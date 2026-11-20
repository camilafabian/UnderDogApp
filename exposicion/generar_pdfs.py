"""Convierte guion.md y estudio_general.md en PDF (guion.pdf y estudio_general.pdf).

    .venv/Scripts/pip install markdown playwright
    .venv/Scripts/python exposicion/generar_pdfs.py

Usa el Edge (o Chrome) instalado en la compu para imprimir; no descarga ningún navegador.
"""

from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

CARPETA = Path(__file__).resolve().parent

ESTILO = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Inter, Arial, sans-serif; color: #1e2430; font-size: 11.5pt; line-height: 1.55; }
h1 { font-family: Georgia, serif; font-size: 24pt; margin: 0 0 4pt; color: #0f1b24; }
h2 { font-family: Georgia, serif; font-size: 16pt; margin: 18pt 0 6pt; color: #0f7f73; border-bottom: 2px solid #0f7f73; padding-bottom: 3pt; }
h1 + p, h1 + p + p { color: #5f6876; margin: 2pt 0; }
p { margin: 6pt 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 10pt; page-break-inside: avoid; }
th, td { text-align: left; padding: 5pt 7pt; border-bottom: 1px solid #e0ddd5; vertical-align: top; }
th { background: #f3f1ec; color: #5f6876; }
blockquote { margin: 10pt 0 6pt; padding: 6pt 10pt; background: #e6eef8; border-left: 4px solid #2563a8; border-radius: 4px; font-size: 10pt; }
blockquote p { margin: 0; }
code { background: #f0eee8; padding: 0 4px; border-radius: 3px; font-size: 9.5pt; }
ul, ol { margin: 4pt 0 4pt 18pt; padding: 0; } li { margin: 3pt 0; }
.meta { display: inline-block; background: #0f1b24; color: #fff; padding: 3pt 10pt; border-radius: 99px; font-size: 9.5pt; margin: 0 0 6pt; }
.pase { margin: 12pt 0 0; padding: 7pt 10pt; background: #e3f3f0; border-radius: 6px; font-weight: 600; color: #0f7f73; }
.resumen { background: #0f1b24; color: #e9eef1; padding: 10pt 14pt; border-radius: 8px; margin: 12pt 0; }
.resumen strong { color: #fff; }
.salto { page-break-after: always; }
"""

# el guion se lee en voz alta: letra más grande
ESTILO_GUION = "body { font-size: 13pt; line-height: 1.65; } blockquote { font-size: 11pt; }"


def a_pdf(nombre: str, extra: str = "") -> Path:
    texto = (CARPETA / f"{nombre}.md").read_text(encoding="utf-8")
    cuerpo = markdown.markdown(texto, extensions=["tables", "md_in_html"])
    html = f"<!doctype html><html lang='es'><head><meta charset='utf-8'><style>{ESTILO}{extra}</style></head><body>{cuerpo}</body></html>"
    salida = CARPETA / f"{nombre}.pdf"
    with sync_playwright() as p:
        try:
            navegador = p.chromium.launch(channel="msedge")
        except Exception:
            navegador = p.chromium.launch(channel="chrome")
        pagina = navegador.new_page()
        pagina.set_content(html, wait_until="load")
        pagina.pdf(path=str(salida), format="A4", print_background=True, display_header_footer=True,
                   header_template="<span></span>",
                   footer_template="<div style='font-size:8pt;color:#888;width:100%;text-align:center'>"
                                   "¿Gana el underdog? · <span class='pageNumber'></span> / <span class='totalPages'></span></div>",
                   margin={"top": "16mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
        navegador.close()
    return salida


if __name__ == "__main__":
    print(a_pdf("guion", ESTILO_GUION))
    print(a_pdf("estudio_general"))
