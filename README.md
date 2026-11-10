# Proyecto final · Cobertura de tesorería y financiamiento de The Coca-Cola Company (KO)

**Derivados y Administración de Riesgos Corporativos (Gpo 40)** · EGADE Business School · PhD. Raymundo Díaz Robles

**Equipo:** Carmen Abigail Cázares Báez (A01452144) · Diana Lorena Lozada Báez (A01452169) · Edgar Jiménez Téllez (A01304961)

Estrategia de financiamiento y cobertura para Coca-Cola basada en el caso Molson Coors (2007): refinanciamiento de 3,000 millones de dólares de bonos que vencen entre marzo y junio de 2027 con un bono convertible, un call spread y un T-lock, y cobertura de los riesgos de tasa, tipo de cambio, commodities y precio de la acción. Las coberturas se implementaron en el simulador de Investopedia con ETFs y opciones.

## Resultado principal

| Concepto | Valor |
|---|---|
| Cupón justo del convertible (prima de 25 %) | 2.71 % |
| Rendimiento de la deuda recta (Tesoro 5 años + CDS) | 5.02 % |
| Ahorro anual sobre 3,000 millones | 69.5 millones USD |
| Costo neto del call spread (prima efectiva de 60 %) | 160.8 millones USD (5.4 % de la emisión) |
| Años de ahorro para pagar el call spread | 2.3 |
| Probabilidad de conversión (Monte Carlo) | 33 % |
| Resultado del portafolio de cobertura en Investopedia | −4,540 USD |

**Recomendación:** emitir el convertible por 3,000 millones con prima de 25 % y cupón de 2.75 %, comprar el call spread a 60 % y fijar la tasa con un T-lock durante la colocación.

## Entregables

| Carpeta | Contenido |
|---|---|
| [`01_Reporte_Ejecutivo`](https://github.com/a01452144-ccb/Proyecto-Final-Derivados-KO-/tree/main/01_Reporte_Ejecutivo) | Reporte ejecutivo: diagnóstico, estrategia, valoración, riesgos y recomendación |
| [`02_Dashboard`](https://github.com/a01452144-ccb/Proyecto-Final-Derivados-KO-/tree/main/02_Dashboard) | Dashboard HTML con Plotly: empresa, deuda, calificaciones, valuación, call spread, escenarios interactivos, Monte Carlo animado y resultados del simulador |
| [`03_Codigo y Datos`](https://github.com/a01452144-ccb/Proyecto-Final-Derivados-KO-/tree/main/03_Codigo%20y%20Datos) | R Markdown y su HTML con el código visible; código en Python (`.py` y notebook con outputs) en `05_Python`; archivos de datos en `datos` |
| [`04_Calculadora`](https://github.com/a01452144-ccb/Proyecto-Final-Derivados-KO-/tree/main/04_Calculadora) | Calculadora interactiva del convertible y el call spread (archivo HTML autocontenido) |
| [`05_Prompt_IA`](https://github.com/a01452144-ccb/Proyecto-Final-Derivados-KO-/tree/main/05_Prompt_IA) | Uso de inteligencia artificial: prompts, código recibido y correcciones |
| [`06_Capturas`](https://github.com/a01452144-ccb/Proyecto-Final-Derivados-KO-/tree/main/06_Capturas) | Capturas de Investopedia y Bloomberg (OMON, DES, FA, DDIS, CRPR, USGG5YR y SWPM) con su explicación |

### Ver en línea

| Entregable | Enlace | Respaldo |
|---|---|---|
| Dashboard interactivo | [Abrir dashboard](https://a01452144-ccb.github.io/Proyecto-Final-Derivados-KO-/02_Dashboard/KO_Dashboard.html) | [Ver aquí](https://raw.githack.com/a01452144-ccb/Proyecto-Final-Derivados-KO-/main/02_Dashboard/KO_Dashboard.html) |
| Calculadora del convertible | [Abrir calculadora](https://a01452144-ccb.github.io/Proyecto-Final-Derivados-KO-/04_Calculadora/Calculadora_Convertible_KO.html) | [Ver aquí](https://raw.githack.com/a01452144-ccb/Proyecto-Final-Derivados-KO-/main/04_Calculadora/Calculadora_Convertible_KO.html) |
| Reporte reproducible (HTML del R Markdown) | [Abrir reporte](https://a01452144-ccb.github.io/Proyecto-Final-Derivados-KO-/03_Codigo%20y%20Datos/Proyecto_Final_KO.html) | [Ver aquí](https://raw.githack.com/a01452144-ccb/Proyecto-Final-Derivados-KO-/main/03_Codigo%20y%20Datos/Proyecto_Final_KO.html) |
| Notebook de Python | [Abrir en Google Colab](https://colab.research.google.com/github/a01452144-ccb/Proyecto-Final-Derivados-KO-/blob/main/03_Codigo%20y%20Datos/05_Python/Valuacion_Convertible_KO.ipynb) | [Copia en Google Drive](https://colab.research.google.com/drive/1FT61K311Yn0GtkBNkJtZOYHdaaIpc5lN?usp=sharing) |

## Datos usados y su fuente

Todos los archivos están en [`03_Codigo y Datos/datos`](https://github.com/a01452144-ccb/Proyecto-Final-Derivados-KO-/tree/main/03_Codigo%20y%20Datos/datos).

| Archivo | Contenido | Fuente |
|---|---|---|
| `parametros_mercado.csv` | Precio, volatilidades, Tesoro, swap, DV01, CDS, dividendo y otros parámetros, con su fuente y fecha | Bloomberg y LSEG |
| `bonos_lseg_ko.csv` | 48 bonos vigentes: vencimiento, monto, cupón y moneda | LSEG Debt Structure, 3-nov-2026 |
| `bloomberg_omon_ko_ene2029.csv` | Volatilidad implícita por strike, opciones a enero de 2029 | Bloomberg OMON, 9-nov-2026 |
| `bloomberg_FA_KO.xlsx` y `.pdf` | Estados financieros trimestrales ajustados | Bloomberg FA, 9-nov-2026 |
| `operaciones_investopedia.csv` | Entradas, salidas y P&L de cada operación | Investopedia Simulator |
| `investopedia_trade_history_2026-11-09.xls` | Exportación original del Trade History | Investopedia Simulator |

Otras fuentes: The Coca-Cola Company, Form 10-K 2025 y 8-K del tercer trimestre de 2026 (SEC); precios históricos de KO de Yahoo Finance (`quantmod` en R y `yfinance` en Python); caso *Molson Coors Brewing Co.: Valuing a Convertible Bond*.

## Metodología

- **Volatilidad:** histórica, EWMA (λ = 0.94) y GARCH(1,1) por máxima verosimilitud, comparadas con la volatilidad implícita de Bloomberg.
- **Valuación del convertible:** bono recto más opción call, con Black-Scholes y árbol binomial CRR (europeo y americano).
- **Call spread:** calls compradas al precio de conversión y warrants vendidos con 60 % de prima, cada pierna con la volatilidad implícita de su strike.
- **Griegas y escenarios:** delta, gamma y vega del call spread; mapa de valor por precio y volatilidad.
- **Monte Carlo:** 20,000 trayectorias de KO a cinco años para medir la probabilidad de conversión y el costo de la dilución.
- **Cobertura de tasa:** T-lock dimensionado con el DV01 de Bloomberg SWPM y comparado con un swap SOFR con inicio diferido.

## Cómo reproducirlo

- **R:** abrir `03_Codigo y Datos/Proyecto_Final_KO.Rmd` en RStudio (paquetes `knitr`, `rmarkdown` y `quantmod`) y dar *Knit*. La carpeta `datos` debe quedar junto al Rmd.
- **Python:** abrir el notebook en Google Colab con el enlace de arriba y ejecutar *Entorno de ejecución → Ejecutar todas*.
- **Dashboard y calculadora:** abrirlos con los enlaces de la sección *Ver en línea*, o descargar los archivos `.html` y abrirlos en cualquier navegador (el dashboard requiere internet para cargar Plotly).

Material académico; no es una recomendación de inversión.
