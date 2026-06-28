# Spanish and Basque prompts

This file documents the current prompt templates implemented in
`src/medical_rag_thesis/prompts.py`.

The pipeline uses one prompt style: `extractive`.

Placeholders:

- `{context}`: retrieved documents, when RAG is enabled.
- `{question}`: merged runtime input from `topic`, `question`, and `subquestion`.
- `{options}`: answer options, when available.
- `{examples}`: few-shot examples, when enabled.
- `{answer}`: first-pass answer in the self-feedback step.

## Spanish

### System prompt

```text
Eres un experto clínico.
```

### Initial prompt with retrieved context

```text
Tu tarea es responder usando la información del contexto recuperado, sin inventar datos.

Reglas:
- Basa tu respuesta en la información del contexto recuperado. NO añadas información externa. COPIA frases exactas cuando sea posible.
- Responde en español.
- Responde SIEMPRE en este formato:

Respuesta corta:
(texto extraído)

Evidencia:
(texto extraído)

{examples}

Contexto recuperado:
{context}

Pregunta:
{question}

Opciones:
{options}
```

### Initial prompt without retrieved context

```text
Tu tarea es responder la pregunta clínica de forma justificada.

Reglas:
- Usa la información de la pregunta (y las opciones de respuesta, si las hay).
- Puedes apoyarte en conocimiento clínico general.
- NO inventes datos concretos que no puedas justificar.
- Responde en español.
- Responde SIEMPRE en este formato:

Respuesta corta:
(respuesta breve)

Evidencia:
(justificación)

{examples}

Pregunta:
{question}

Opciones:
{options}
```

### Self-feedback prompt with retrieved context

```text
Revisa la siguiente respuesta.

Comprueba:
- si está basada en el contexto recuperado.
- si hay alucinaciones.
- si falta información.

Reescribe la respuesta mejorada.

Reglas obligatorias:
- Responde SOLO con los dos campos indicados: "Respuesta corta" y "Evidencia".
- NO repitas las instrucciones.
- Responde en español.
- Usa SOLO información del contexto recuperado.

Responde SIEMPRE en este formato:

Respuesta corta:
(texto extraído)

Evidencia:
(texto extraído)

Contexto recuperado:
{context}

Pregunta:
{question}

Opciones:
{options}

Respuesta original:
{answer}

Escribe ahora únicamente la respuesta final con los dos campos indicados.
```

### Self-feedback prompt without retrieved context

```text
Revisa la siguiente respuesta.

Comprueba:
- si responde a la pregunta.
- si hay alucinaciones.
- si falta información relevante.

Reescribe la respuesta mejorada.

Reglas obligatorias:
- Responde SOLO con los dos campos indicados: "Respuesta corta" y "Evidencia".
- NO repitas las instrucciones.
- Responde en español.
- Usa la pregunta, las opciones de respuesta (si existen), y conocimiento clínico general.

Responde SIEMPRE en este formato:

Respuesta corta:
(respuesta breve)

Evidencia:
(justificación)

Pregunta:
{question}

Opciones:
{options}

Respuesta original:
{answer}

Escribe ahora únicamente la respuesta final con los dos campos indicados.
```

## Basque

### System prompt

```text
Aditu klinikoa zara.
```

### Initial prompt with retrieved context

```text
Zure zeregina errekuperatutako testuinguruko informazioa erabiliz erantzutea da, daturik asmatu gabe.

Arauak:
- Oinarritu erantzuna errekuperatutako testuinguruko informazioan. EZ gehitu kanpoko informaziorik. KOPIATU esaldi osoak posible denean.
- Erantzun euskaraz.
- Erantzun BETI honako formatuan:
Erantzun laburra:
(erauztitako testua)

Ebidentzia:
(erauztitako testua)

{examples}

Errekuperatutako testuingurua:
{context}

Galdera:
{question}

Aukerak:
{options}
```

### Initial prompt without retrieved context

```text
Zure zeregina galdera klinikoa modu justifikatuan erantzutea da.

Arauak:
- Erabili galderaren informazioa (eta erantzun-aukerak, egonez gero).
- Medikuntzako ezagutza orokorraz baliatu zaitezke.
- EZ asmatu justifika ezin duzun datu zehatzik.
- Erantzun euskaraz.
- Erantzun BETI honako formatuan:

Erantzun laburra:
(erantzun laburra)

Ebidentzia:
(ebidentzia edo justifikazioa)

{examples}

Galdera:
{question}

Aukerak:
{options}
```

### Self-feedback prompt with retrieved context

```text
Ondorengo erantzuna berrikusi.

Egiaztatu:
- errekuperatutako testuinguruan oinarritua dagoen.
- aluzinazioak dauden.
- informazioa falta den.

Erantzun hobetua berridatzi.

Arau derrigorrezkoak:
- Erantzun SOILIK bi eremu hauekin: "Erantzun laburra" eta "Ebidentzia".
- EZ errepikatu argibideak.
- Erantzun euskaraz.
- Erabili SOILIK errekuperatutako testuingurutik datorren informazioa.

Erantzun BETI honako formatuan:

Erantzun laburra:
(erauztitako testua)

Ebidentzia:
(erauztitako testua)

Errekuperatutako testuingurua:
{context}

Galdera:
{question}

Aukerak:
{options}

Jatorrizko erantzuna:
{answer}

Idatzi orain azken erantzuna soilik, adierazitako bi eremuekin.
```

### Self-feedback prompt without retrieved context

```text
Ondorengo erantzuna berrikusi.

Egiaztatu:
- galderari erantzuten dion.
- aluzinazioak dauden.
- informazio garrantzitsua falta den.

Erantzun hobetua berridatzi.

Arau derrigorrezkoak:
- Erantzun SOILIK bi eremu hauekin: "Erantzun laburra" eta "Ebidentzia".
- EZ errepikatu argibideak.
- Erantzun euskaraz.
- Erabili galdera, erantzun-aukerak (egonez gero), eta medikuntzako ezagutza orokorra.

Erantzun BETI honako formatuan:

Erantzun laburra:
(erantzun laburra)

Ebidentzia:
(ebidentzia edo justifikazioa)

Galdera:
{question}

Aukerak:
{options}

Jatorrizko erantzuna:
{answer}

Idatzi orain azken erantzuna soilik, adierazitako bi eremuekin.
```
