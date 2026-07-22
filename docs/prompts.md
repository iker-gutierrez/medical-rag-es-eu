# Spanish and Basque prompts

This file documents the current prompt templates implemented in
`src/medical_rag_thesis/prompts.py`.

This is the v2 wording: answers are based on the retrieved context, with
exact copying asked only when the context is directly related to the
question, and an explicit fallback to the model's medical knowledge when the
context is insufficient. The previous strict-extractive wording (copy always,
no external knowledge) is preserved in `prompts_v1.md`.

The pipeline uses one prompt style: `extractive`.

Each language/context combination uses a single template. One rule and one
line of the output format are conditional on whether the record has multiple-
choice `options` (CasiMedicos) or not (SNS1064, open-answer):

- The rule "Si hay opciones de respuesta, en el apartado de 'Respuesta corta'
  incluye solo el número de la opción elegida (ej. '3.') y el texto de esa
  opción." (ES) / the Basque equivalent below is always present in the
  instructions, worded as conditional ("si hay..."/"badaude...").
- The `{short_answer}` changes to show the expected
  `(número de opción. ...)` / `(aukeraren zenbakia. ...)` shape only when the
  record actually has options; for open-answer records it stays a plain
  `(respuesta breve)` / `(erantzun laburra)`.
- The `{options}` block itself (the numbered choices, with no header label)
  is only appended when `options` is non-empty.

Variables (resolved by the code before the prompt reaches the model; the
literal `{...}` text never appears in what the model sees):

- `{context}`: retrieved documents, when RAG is enabled.
- `{question}`: merged runtime input from `topic`, `question`, and `subquestion`.
- `{options}`: answer options, only rendered for multiple-choice records.
- `{examples}`: few-shot examples, when enabled. Each renders as a numbered
  block (`Ejemplo N` / `Adibidea N`) containing the example's question,
  options (if any), and its gold answer with each field label on its own
  line (`Respuesta corta:` / `Evidencia:` followed by the content on the
  next line), matching the requested output format exactly.
- `{initial_answer}`: first-pass answer in the self-feedback step.
- `{short_answer}`: resolves to one of the `<...>` format placeholders below
  (`<número de opción. ...>` for multiple-choice records, the plain
  open-answer placeholder otherwise), depending on whether the record has
  options.

Format placeholders (the opposite kind: the model receives these `<...>`
markers literally, as part of the prompt, and is instructed to replace them
with real content rather than reproduce them verbatim -- see the closing
"Los símbolos < > ..." sentence in each prompt):

- `<respuesta breve>` / `<erantzun laburra>`: expected shape of an
  open-answer short answer.
- `<número de opción. texto de opción>` / `<aukeraren zenbakia. aukeraren
  testua>`: expected shape for multiple-choice records.
- `<evidencia basada en el contexto extraído...>` / the Basque
  equivalent: expected shape of the Evidencia field in with-context prompts.
- `<justificación>` / `<ebidentzia edo justifikazioa>`: expected shape of
  the Evidencia field in no-context prompts.

## Spanish

### System prompt

```text
Eres un experto médico.
```

### Initial prompt with retrieved context

```text
Tu tarea es responder a la pregunta médica de forma justificada.

Reglas:
- Basa tu respuesta en la información del contexto extraído. Puedes copiar frases exactas cuando el contexto extraído esté directamente relacionado con la pregunta. Si el contexto extraído es insuficiente para una respuesta completa, puedes combinar tu conocimiento médico con el contexto extraído.
- NO inventes datos que no estén respaldados por el contexto extraído o por tu conocimiento médico.
- Si la pregunta tiene opciones de respuesta, en el apartado de "Respuesta corta" incluye solo el índice de la opción elegida (ej. "3.") y el texto de esa opción. Si la pregunta no tiene opciones de respuesta, en el apartado de "Respuesta corta" responde directamente con tus palabras.
- Responde en español.

{examples}

Contexto extraído:
{context}

Pregunta:
{question}

{options}  # block only present when the record has options

Responde en este formato, sin incluir texto fuera de los campos indicados:

Respuesta corta:
{short_answer}  # "<número de opción. texto de opción>" if options, else "<respuesta breve generada>"

Evidencia:
<evidencia basada en el contexto extraído y, si también lo has usado, en tu conocimiento médico>

Los símbolos < > delimitan marcadores de posición que especifican el tipo de contenido esperado. En tu respuesta, sustituye esos marcadores de posición y sus símbolos delimitadores por el contenido esperado.
```

### Initial prompt without retrieved context

```text
Tu tarea es responder a la pregunta médica de forma justificada.

Reglas:
- Usa la información de la pregunta (y las opciones de respuesta, si las hay).
- Apóyate en tu conocimiento médico.
- NO inventes datos que no puedas justificar con tu conocimiento médico.
- Si la pregunta tiene opciones de respuesta, en el apartado de "Respuesta corta" incluye solo el índice de la opción elegida (ej. "3.") y el texto de esa opción. Si la pregunta no tiene opciones de respuesta, en el apartado de "Respuesta corta" responde directamente con tus palabras.
- Responde en español.

{examples}

Pregunta:
{question}

{options}  # block only present when the record has options

Responde en este formato, sin incluir texto fuera de los campos indicados:

Respuesta corta:
{short_answer}  # "<número de opción. texto de opción>" if options, else "<respuesta breve generada>"

Evidencia:
<evidencia basada en tu conocimiento médico>

Los símbolos < > delimitan marcadores de posición que especifican el tipo de contenido esperado. En tu respuesta, sustituye esos marcadores de posición y sus símbolos delimitadores por el contenido esperado.
```

### Self-feedback prompt with retrieved context

```text
Revisa la siguiente respuesta.

Comprueba:
- si está basada en el contexto extraído (y en tu conocimiento médico, si lo has usado).
- si hay alucinaciones.
- si falta información.

Reescribe una respuesta mejorada. Reglas:
- Basa tu respuesta en la información del contexto extraído. Puedes copiar frases exactas cuando el contexto extraído esté directamente relacionado con la pregunta. Si el contexto extraído es insuficiente para una respuesta completa, puedes combinar tu conocimiento médico con el contexto extraído.
- NO inventes datos que no estén respaldados por el contexto extraído o por tu conocimiento médico.
- Si la pregunta tiene opciones de respuesta, en el apartado de "Respuesta corta" incluye solo el índice de la opción elegida (ej. "3.") y el texto de esa opción. Si la pregunta no tiene opciones de respuesta, en el apartado de "Respuesta corta" responde directamente con tus palabras.
- Responde en español.

Contexto extraído:
{context}

Pregunta:
{question}

{options}  # block only present when the record has options

Respuesta inicial:
{initial_answer}

Escribe únicamente la respuesta mejorada. Responde en este formato, sin incluir texto fuera de los campos indicados:

Respuesta corta:
{short_answer}  # "<número de opción. texto de opción>" if options, else "<respuesta breve generada>"

Evidencia:
<evidencia basada en el contexto extraído y, si también lo has usado, en tu conocimiento médico>

Los símbolos < > delimitan marcadores de posición que especifican el tipo de contenido esperado. En tu respuesta, sustituye esos marcadores de posición y sus símbolos delimitadores por el contenido esperado.
```

### Self-feedback prompt without retrieved context

```text
Revisa la siguiente respuesta.

Comprueba:
- si está basada en tu conocimiento médico.
- si hay alucinaciones.
- si falta información relevante.

Reescribe una respuesta mejorada. Reglas:
- Usa la información de la pregunta (y las opciones de respuesta, si las hay).
- Apóyate en tu conocimiento médico.
- NO inventes datos que no puedas justificar con tu conocimiento médico.
- Si la pregunta tiene opciones de respuesta, en el apartado de "Respuesta corta" incluye solo el índice de la opción elegida (ej. "3.") y el texto de esa opción. Si la pregunta no tiene opciones de respuesta, en el apartado de "Respuesta corta" responde directamente con tus palabras.
- Responde en español.

Pregunta:
{question}

{options}  # block only present when the record has options

Respuesta inicial:
{initial_answer}

Escribe únicamente la respuesta mejorada. Responde en este formato, sin incluir texto fuera de los campos indicados:

Respuesta corta:
{short_answer}  # "<número de opción. texto de opción>" if options, else "<respuesta breve generada>"

Evidencia:
<evidencia basada en tu conocimiento médico>

Los símbolos < > delimitan marcadores de posición que especifican el tipo de contenido esperado. En tu respuesta, sustituye esos marcadores de posición y sus símbolos delimitadores por el contenido esperado.
```

## Basque

### System prompt

```text
Aditu medikoa zara.
```

### Initial prompt with retrieved context

```text
Zure ataza galdera medikoari modu justifikatuan erantzutea da.

Arauak:
- Oinarritu erantzuna erauzitako testuinguruko informazioan. Kopiatu ditzakezu esaldi osoak erauzitako testuingurua galderarekin zuzenki erlazionatuta dagoenean. Erauzitako testuingurua erantzun oso baterako nahikoa ez bada, zure medikuntza-ezagutza erauzitako testuinguruarekin konbina dezakezu.
- EZ asmatu erauzitako testuinguruan edo zure medikuntza-ezagutzan oinarrituta ez dagoen daturik.
- Galderak erantzun-aukerak baditu, "Erantzun laburra" atalean sartu soilik aukeratutako aukeraren indizea (adib. "3.") eta aukera horren testua. Galderak erantzun-aukerarik ez badu, "Erantzun laburra" atalean erantzun zuzenean zure hitzekin.
- Erantzun euskaraz.

{examples}

Erauzitako testuingurua:
{context}

Galdera:
{question}

{options}  # block only present when the record has options

Erantzun honako formatuan, testurik gehitu gabe adierazitako eremuetatik kanpo:

Erantzun laburra:
{short_answer}  # "<número de opción. texto de opción>" if options, else "<respuesta breve generada>"

Ebidentzia:
<erauzitako testuinguruan oinarritutako ebidentzia eta, erabili baduzu, zure medikuntza-ezagutzan oinarritutakoa ere>

< > sinboloek espero den eduki-mota adierazten duten leku-markak mugatzen dituzte. Zure erantzunean, ordezkatu leku-marka horiek eta haien muga-sinboloak espero den edukiarekin.
```

### Initial prompt without retrieved context

```text
Zure ataza galdera medikoari modu justifikatuan erantzutea da.

Arauak:
- Erabili galderaren informazioa (eta erantzun-aukerak, egonez gero).
- Baliatu zure medikuntza-ezagutzaz.
- EZ asmatu zure medikuntza-ezagutzarekin justifikatu ezin duzun daturik.
- Galderak erantzun-aukerak baditu, "Erantzun laburra" atalean sartu soilik aukeratutako aukeraren indizea (adib. "3.") eta aukera horren testua. Galderak erantzun-aukerarik ez badu, "Erantzun laburra" atalean erantzun zuzenean zure hitzekin.
- Erantzun euskaraz.

{examples}

Galdera:
{question}

{options}  # block only present when the record has options

Erantzun honako formatuan, testurik gehitu gabe adierazitako eremuetatik kanpo:

Erantzun laburra:
{short_answer}  # "<número de opción. texto de opción>" if options, else "<respuesta breve generada>"

Ebidentzia:
<zure medikuntza-ezagutzan oinarritutako ebidentzia>

< > sinboloek espero den eduki-mota adierazten duten leku-markak mugatzen dituzte. Zure erantzunean, ordezkatu leku-marka horiek eta haien muga-sinboloak espero den edukiarekin.
```

### Self-feedback prompt with retrieved context

```text
Ondorengo erantzuna berrikusi.

Egiaztatu:
- erauzitako testuinguruan oinarritua dagoen (eta zure medikuntza-ezagutzan, erabili baduzu).
- aluzinazioak dauden.
- informazioa falta den.

Berridatzi erantzun hobetu bat. Arauak:
- Oinarritu erantzuna erauzitako testuinguruko informazioan. Kopiatu ditzakezu esaldi osoak erauzitako testuingurua galderarekin zuzenki erlazionatuta dagoenean. Erauzitako testuingurua erantzun oso baterako nahikoa ez bada, zure medikuntza-ezagutza erauzitako testuinguruarekin konbina dezakezu.
- EZ asmatu erauzitako testuinguruak edo zure medikuntza-ezagutzak babesten ez duen daturik.
- Galderak erantzun-aukerak baditu, "Erantzun laburra" atalean sartu soilik aukeratutako aukeraren indizea (adib. "3.") eta aukera horren testua. Galderak erantzun-aukerarik ez badu, "Erantzun laburra" atalean erantzun zuzenean zure hitzekin.
- Erantzun euskaraz.

Erauzitako testuingurua:
{context}

Galdera:
{question}

{options}  # block only present when the record has options

Hasierako erantzuna:
{initial_answer}

Idatzi soilik erantzun hobetua. Erantzun honako formatuan, testurik gehitu gabe adierazitako eremuetatik kanpo:

Erantzun laburra:
{short_answer}  # "<número de opción. texto de opción>" if options, else "<respuesta breve generada>"

Ebidentzia:
<erauzitako testuinguruan oinarritutako ebidentzia eta, erabili baduzu, zure medikuntza-ezagutzan oinarritutakoa ere>

< > sinboloek espero den eduki-mota adierazten duten leku-markak mugatzen dituzte. Zure erantzunean, ordezkatu leku-marka horiek eta haien muga-sinboloak espero den edukiarekin.
```

### Self-feedback prompt without retrieved context

```text
Ondorengo erantzuna berrikusi.

Egiaztatu:
- zure medikuntza-ezagutzan oinarrituta dagoen.
- aluzinazioak dauden.
- informazio garrantzitsua falta den.

Berridatzi erantzun hobetu bat. Arauak:
- Erabili galderaren informazioa (eta erantzun-aukerak, egonez gero).
- Baliatu zure medikuntza-ezagutzaz.
- EZ asmatu zure medikuntza-ezagutzarekin justifikatu ezin duzun daturik.
- Galderak erantzun-aukerak baditu, "Erantzun laburra" atalean sartu soilik aukeratutako aukeraren indizea (adib. "3.") eta aukera horren testua. Galderak erantzun-aukerarik ez badu, "Erantzun laburra" atalean erantzun zuzenean zure hitzekin.
- Erantzun euskaraz.

Galdera:
{question}

{options}  # block only present when the record has options

Hasierako erantzuna:
{initial_answer}

Idatzi soilik erantzun hobetua. Erantzun honako formatuan, testurik gehitu gabe adierazitako eremuetatik kanpo:

Erantzun laburra:
{short_answer}  # "<número de opción. texto de opción>" if options, else "<respuesta breve generada>"

Ebidentzia:
<zure medikuntza-ezagutzan oinarritutako ebidentzia>

< > sinboloek espero den eduki-mota adierazten duten leku-markak mugatzen dituzte. Zure erantzunean, ordezkatu leku-marka horiek eta haien muga-sinboloak espero den edukiarekin.
```
