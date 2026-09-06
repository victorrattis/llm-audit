# llm-audit para Claude Code

Auditoria por turno do Claude Code, no mesmo formato que o
`.opencode/plugins/llm-audit.ts` escreve para o opencode — uma pasta por
conversa, com `turn_NNN.json`, `turns.jsonl`, `summary.json` e `state.json`, para
que o mesmo `jq` leia os dois lados.

O Claude Code é fechado e não embrulha nenhum `fetch`, então aqui não há captura:
o que existe é um leitor incremental do que o próprio Claude Code já grava em
disco.

## Como funciona

Duas fontes, casadas por turno — sendo um turno **uma requisição ao modelo**, que
é o que o `requestId` nomeia:

| fonte | o que dá |
|---|---|
| transcript `~/.claude/projects/<slug>/<sessão>.jsonl` | a conta exata do provider (`message.usage`), modelo, effort, e a resposta **com o thinking íntegro** |
| raw bodies de `OTEL_LOG_RAW_API_BODIES=file:<dir>` | a requisição Messages API não truncada: system prompt, schemas das tools, todas as mensagens, `cache_control`, opções de sampling |

O transcript sozinho já produz o turno. Os bodies são o que acrescenta o prompt
decomposto e o diff de reuso; sem eles o turno sai com `prompt` ausente e um
`prompt_note` dizendo por quê.

### A armadilha do transcript

Uma resposta da API é persistida como **várias entradas**, uma por bloco de
conteúdo, todas repetindo o mesmo `message.id`, o mesmo `requestId` e **o mesmo
objeto `usage`**. Numa sessão real: 372 entradas `assistant` para 189 requisições.
Somar `usage` por entrada triplicaria a conta. Por isso o agrupamento é por
`requestId` e a conta é contada uma vez por grupo.

## Arquivos

| arquivo | o que é |
|---|---|
| `core.ts` | o núcleo do `llm-audit.ts` do opencode, verbatim menos a camada de captura (fetch, decodificação HTTP, redação). São as mesmas funções e os mesmos tipos — é isso que garante que a saída tenha o mesmo formato dos dois lados |
| `report.ts` | o leitor: lê o transcript, casa com os bodies, escreve as pastas |
| `prices.json` | catálogo de preços por milhão de tokens, para o campo `cost` |

Saída em `CLAUDE_LLM_AUDIT_DIR` (default
`~/.local/share/claude-code/log/llm-audit`), uma pasta
`<AAAAMMDD_HHMMSS>-<sessão>` por conversa. Os corpos brutos ficam **ao lado**, em
`<dir>-bodies`, nunca dentro: o viewer varre recursivamente tudo que termina em
`.json`, e cada corpo tem ~1 MB. Apontar `OTEL_LOG_RAW_API_BODIES` para o próprio
diretório de auditoria é o erro fácil de cometer, então cada passagem varre os
`*.request.json` / `*.response.json` soltos na raiz do diretório para
`<dir>-bodies` antes de ler — o layout se conserta sozinho, e é isso que permite
mudar o caminho sem reiniciar o Claude Code.

## Configuração

`~/.claude/settings.json` faz as duas metades: o `env` liga a gravação dos corpos
(variáveis de telemetria aplicam no startup, de qualquer arquivo de settings), e
os hooks `Stop`, `SubagentStop` e `SessionEnd` chamam o leitor. `Stop` roda
`async` para não somar latência ao turno; `SessionEnd` roda síncrono para
garantir o `summary.json` final.

Variáveis que o `report.ts` lê:

| variável | efeito |
|---|---|
| `CLAUDE_LLM_AUDIT_DIR` | onde escrever (default `~/.local/share/claude-code/log/llm-audit`) |
| `CLAUDE_LLM_AUDIT_BODIES` | onde estão os corpos. Sem ela, tenta `OTEL_LOG_RAW_API_BODIES`, depois `<dir>-bodies`, por último `<dir>/bodies` — necessário porque o Claude Code só exporta para subprocessos as variáveis de telemetria que classifica como "safe", e `OTEL_LOG_RAW_API_BODIES` não é uma delas: um hook não a enxerga. Por isso os comandos dos hooks trazem `CLAUDE_LLM_AUDIT_DIR` e `CLAUDE_LLM_AUDIT_BODIES` como prefixo, que não depende do que o `env` do settings repassa |
| `CLAUDE_LLM_AUDIT=0` | desliga |
| `CLAUDE_LLM_AUDIT_INLINE=0` | aponta para o corpo em disco em vez de copiá-lo para dentro do turno. Pastas bem menores, mas **o viewer para de funcionar** — veja abaixo |

`bun report.ts --prune 7` apaga corpos com mais de 7 dias — é a parte que cresce
sem limite.

## O que o viewer exige

O `harness-audit-viewer` bifurca em `raw.request.body`. Com ele, usa os campos já
decodificados pelo plugin — prompt, usage, cost, reply — e é o caminho bom. **Sem
ele, cai num ramo legado que lê `raw.request` como se fosse o próprio corpo**, e
o turno inteiro sai zerado: sem modelo, sem tokens, sem conversa.

Por isso o corpo é embutido por padrão, apesar de custar ~1 MB por turno (uma
sessão de 67 turnos dá ~27 MB). `CLAUDE_LLM_AUDIT_INLINE=0` troca isso por um
ponteiro, e aí só o `jq` funciona.

Um turno sem corpo casado — os anteriores a ligar a telemetria — leva em
`request.body` a única coisa que se sabe da requisição pelo transcript, o
`model`. Isso o mantém no ramo bom, com usage, custo e resposta corretos e a
conversa vazia; nunca uma reconstrução de um prompt que não foi observado. Na
tabela do viewer eles aparecem com `MSGS 0` e prefixo `—`.

## Diferenças em relação ao plugin do opencode

Nenhuma delas é escolha de estilo; cada uma é algo que o Claude Code não expõe.

- **`response` não tem `events`.** O Claude Code grava o corpo final, não o
  stream SSE, então o viewer mostra "sem stream no log" e remonta a resposta a
  partir do `reply` — que vem do transcript, com o thinking íntegro.
- **`request.method` é `POST`; `url` e `headers` são `null`.** O corpo é a única
  parte do envelope que o Claude Code grava.
- **`request.match`** diz o quão certo é o casamento da requisição. A resposta
  casa exato — o arquivo se chama pelo `request_id`, que é o `requestId` do
  transcript. A requisição é nomeada por um uuid gerado pelo cliente, que o
  transcript não registra, então: `exact` quando o corpo da resposta nomeia esse
  uuid, `adjacent` quando é a última requisição não reclamada escrita antes da
  resposta, para o mesmo modelo, `none` quando não havia nada.
- **`elapsed_seconds`** não é a duração da requisição — o Claude Code só reporta
  isso por OpenTelemetry, nunca a um hook. É o intervalo entre a entrada ficar
  pronta e o primeiro bloco da resposta aparecer.
- **`runs`** lista as versões do Claude Code que escreveram a pasta, não os
  processos: no opencode um run é um processo, aqui o processo é o app inteiro.
  Não há `runs/` nem `latest.json`.
- **Sem `estimated`.** O transcript sempre traz a conta do provider, então
  `usage.source` é sempre `"provider"` e o palpite de chars/4 nunca entra.
- **Subagentes** não têm id próprio no transcript; a chave é
  `<sessão>#<uuid-do-topo-da-cadeia-sidechain>`, para que dois subagentes
  simultâneos não sejam diferenciados um contra o outro no diff de reuso.
- **Preços vêm do `prices.json` local**, não de um catálogo de provider. Modelo
  fora da tabela sai sem `cost` — nunca com um número inventado.

## O que foi verificado em produção

- **Os corpos são gravados** com `OTEL_LOGS_EXPORTER=console` — confirmado com
  arquivos reais de ~1 MB por requisição. Se `none` também servir, é mais
  silencioso; não foi testado.
- **Não existe join exato requisição↔turno.** Nenhum uuid de arquivo
  `.request.json` aparece em nenhum `.response.json`. O código ainda tenta o
  caminho exato primeiro, mas na prática cai para `adjacent`.
  `jq -r .request_match turns.jsonl | sort | uniq -c` mostra a distribuição.

## Cuidados

- Os corpos contêm **a conversa inteira**: system prompt, todo o histórico, todos
  os resultados de tool. O diretório é material sensível.
- O formato do transcript é **interno ao Claude Code e muda entre versões** — a
  própria documentação marca o join por `requestId` como versionado, não
  contratual. Cada turno grava a `version` que o produziu, para que uma quebra
  futura seja diagnosticável.
- O thinking é sempre redigido dos raw bodies. Por isso a resposta é montada a
  partir do transcript, que o preserva.
- O leitor nunca derruba um turno: qualquer falha vai para `errors.log` na raiz
  do diretório de auditoria e o processo sai com 0.
