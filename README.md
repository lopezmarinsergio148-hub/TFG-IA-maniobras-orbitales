# TFG — Inteligencia artificial para maniobras orbitales

Trabajo de Fin de Grado en Ingeniería Aeroespacial (Universidad de Castilla-La Mancha).
Un sistema que **planifica y explica maniobras orbitales** combinando tres piezas: un modelo
físico validado contra datos de misiones reales, agentes de **aprendizaje por refuerzo (RL)** que
resuelven las maniobras sin solución analítica, y una capa de **lenguaje natural (LLM)** que
orquesta el conjunto en una conversación.

**Autor:** Sergio López Marín

---

## Qué hace

- **Modelo atmosférico de 9 cuerpos** (Tierra, Marte, Venus, Júpiter, Saturno, Urano, Neptuno,
  Luna y Mercurio), con densidades derivadas de fuentes primarias y validadas contra misiones
  (Galileo, Cassini, Voyager 2, MCD, USSA-76).
- **Propagador orbital** con achatamiento (J2) y arrastre atmosférico sobre `poliastro`
  (`CowellPropagator`), verificado frente a la teoría de Brouwer (1959).
- **5 agentes de RL** (PPO / Stable-Baselines3), todos con la física validada:
  1. Transferencia de Hohmann LEO→GEO (a 0,03 % del óptimo analítico).
  2. Transferencias coplanares generales por adimensionalización (invariancia de escala).
  3. Aerofrenado multiplaneta (7 cuerpos), con criterio de peligro por presión dinámica.
  4. Transferencias 3D con cambio de plano (acción vectorial, < 2 % del óptimo).
  5. Mantenimiento orbital / *station-keeping* frente al arrastre.
- **Capa LLM** (Groq / Llama 3.3) con 13 herramientas (cálculo + dibujo) que mapean a todo el
  proyecto mediante *tool use*, anclada a los números reales que devuelven los agentes.

---

## Estructura del repositorio

```
├── ia/                   Agentes de RL (env_/train_/evaluar_/modelo_) + capa LLM (llm_)
├── scripts_importantes/  Código de física optimizado (atmósferas y propagador)
├── datos_validacion/     Datos in-situ de misiones (Galileo, Cassini, Voyager) + validación
├── figuras/              Scripts que generan las figuras del proyecto (matplotlib)
├── imagenes/             Figuras generadas (validaciones y resultados)
├── tests/                Suite de verificación del propagador (54 comprobaciones)
│
├── Orbitas_en_planetas.py     Maniobras de impulsos y visualizaciones (referencia)
└── Orbitas_entre_planetas.py  Hohmann, Lambert y porkchops interplanetarios
```

---

## Instalación

Requiere Python 3.10. Se recomienda un entorno virtual:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

La capa LLM necesita una clave de la API de Groq (gratuita) en la variable de entorno
`GROQ_API_KEY`.

---

## Uso

**Entrenar y evaluar un agente de RL** (ejemplo con el aerofrenado):

```bash
python ia/train_drag.py        # entrena el agente (guarda el mejor modelo)
python ia/evaluar_drag.py      # evalúa el agente frente a la estrategia de referencia
```

**Lanzar el asistente conversacional (capa LLM):**

```bash
python ia/llm_orquestador.py
```

**Ejecutar la suite de verificación del propagador:**

```bash
python tests/test_propagador.py
```

---

## Validación física

El modelo atmosférico se ha contrastado con datos in-situ de misiones reales y la regresión nodal
del propagador se ha verificado frente a la teoría de Brouwer. El detalle está en
`datos_validacion/`.
