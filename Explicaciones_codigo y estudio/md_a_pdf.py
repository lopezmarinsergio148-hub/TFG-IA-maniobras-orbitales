# ═══════════════════════════════════════════════════════════════════════════
#  md_a_pdf.py — Convertidor sencillo de Markdown a PDF (con reportlab)
#
#  Uso:   python md_a_pdf.py archivo.md  [salida.pdf]
#         (si no se da salida, usa el mismo nombre con extensión .pdf)
#
#  Pensado para el flujo: el .md es el MAESTRO editable; este script regenera el
#  PDF cuando haga falta. Soporta el subconjunto de Markdown que usamos:
#    #, ##, ###  encabezados      - listas        --- regla horizontal
#    > cita       **negrita**      `codigo`       | tablas |
#  Usa la fuente DejaVu (de matplotlib) para que se vean bien μ, Δ, ², ×, °, ≤...
# ═══════════════════════════════════════════════════════════════════════════

import os
import re
import sys

import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                HRFlowable, ListFlowable, ListItem, Table, TableStyle)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

# ── Fuente con soporte unicode (DejaVu, que trae matplotlib) ────────────────
_FDIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu",      os.path.join(_FDIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(_FDIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Obl",  os.path.join(_FDIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono",  os.path.join(_FDIR, "DejaVuSansMono.ttf")))
registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                   italic="DejaVu-Obl", boldItalic="DejaVu-Bold")

AZUL = colors.HexColor("#1a3a6e")

ESTILOS = {
    "h1":   ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=17, textColor=AZUL,
                           spaceBefore=6, spaceAfter=12, leading=21),
    "h2":   ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=13.5, textColor=AZUL,
                           spaceBefore=14, spaceAfter=6, leading=17),
    "h3":   ParagraphStyle("h3", fontName="DejaVu-Bold", fontSize=11, textColor=colors.black,
                           spaceBefore=9, spaceAfter=3, leading=14),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9.5, leading=13.5,
                           spaceAfter=4, alignment=4),  # 4 = justificado
    "bullet": ParagraphStyle("bullet", fontName="DejaVu", fontSize=9.5, leading=13.5),
    "quote": ParagraphStyle("quote", fontName="DejaVu-Obl", fontSize=9, leading=12.5,
                            textColor=colors.HexColor("#555555"), leftIndent=10,
                            spaceBefore=4, spaceAfter=6),
}


def inline(texto):
    """Convierte marcas inline (**negrita**, `codigo`) a etiquetas de reportlab."""
    texto = texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    texto = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", texto)
    texto = re.sub(r"`(.+?)`", r'<font face="DejaVuMono" size="8.5">\1</font>', texto)
    return texto


def construir(md_path, pdf_path):
    with open(md_path, encoding="utf-8") as f:
        lineas = f.read().split("\n")

    flow = []
    buf_par, buf_bul, buf_cita, buf_tabla = [], [], [], []

    def vaciar_par():
        if buf_par:
            flow.append(Paragraph(inline(" ".join(buf_par)), ESTILOS["body"]))
            buf_par.clear()

    def vaciar_tabla():
        if not buf_tabla:
            return
        # Trocea cada fila por '|' (ignorando los bordes externos)
        filas = [[c.strip() for c in fila.strip().strip("|").split("|")]
                 for fila in buf_tabla]
        buf_tabla.clear()

        # Descarta la fila separadora de la cabecera (|---|:--:|...)
        def es_separadora(celdas):
            no_vacias = [c for c in celdas if c != ""]
            return bool(no_vacias) and all(re.fullmatch(r":?-{1,}:?", c) for c in no_vacias)
        filas = [f for f in filas if not es_separadora(f)]
        if not filas:
            return

        ncol = max(len(f) for f in filas)
        for f in filas:                       # rellena filas cortas
            f += [""] * (ncol - len(f))

        st_h = ParagraphStyle("celda_h", fontName="DejaVu-Bold", fontSize=8.5,
                              leading=11, textColor=colors.white)
        st_c = ParagraphStyle("celda", fontName="DejaVu", fontSize=8.5, leading=11)
        data = [[Paragraph(inline(c), st_h if i == 0 else st_c) for c in f]
                for i, f in enumerate(filas)]

        ancho = A4[0] - 4 * cm                 # ancho útil (márgenes de 2 cm)
        tabla = Table(data, colWidths=[ancho / ncol] * ncol, repeatRows=1)
        tabla.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), AZUL),                       # cabecera azul
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#eef2f8")]),               # filas alternas
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#aab4c5")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        flow.append(tabla)
        flow.append(Spacer(1, 6))

    def vaciar_bul():
        if buf_bul:
            items = [ListItem(Paragraph(inline(b), ESTILOS["bullet"]), leftIndent=14)
                     for b in buf_bul]
            flow.append(ListFlowable(items, bulletType="bullet", start="•",
                                     bulletFontName="DejaVu", leftIndent=12))
            flow.append(Spacer(1, 4))
            buf_bul.clear()

    def vaciar_cita():
        if buf_cita:
            flow.append(Paragraph(inline(" ".join(buf_cita)), ESTILOS["quote"]))
            buf_cita.clear()

    def vaciar_todo():
        vaciar_par(); vaciar_bul(); vaciar_cita(); vaciar_tabla()

    for raw in lineas:
        s = raw.strip()
        if s == "":
            vaciar_todo(); continue
        if s.startswith("### "):
            vaciar_todo(); flow.append(Paragraph(inline(s[4:]), ESTILOS["h3"]))
        elif s.startswith("## "):
            vaciar_todo(); flow.append(Paragraph(inline(s[3:]), ESTILOS["h2"]))
        elif s.startswith("# "):
            vaciar_todo(); flow.append(Paragraph(inline(s[2:]), ESTILOS["h1"]))
        elif s.startswith("---"):
            vaciar_todo()
            flow.append(HRFlowable(width="100%", thickness=0.6,
                                   color=colors.HexColor("#bbbbbb"),
                                   spaceBefore=6, spaceAfter=8))
        elif s.startswith("|"):
            vaciar_par(); vaciar_bul(); vaciar_cita()
            buf_tabla.append(s)
        elif s.startswith(">"):
            vaciar_par(); vaciar_bul(); vaciar_tabla()
            buf_cita.append(s.lstrip("> ").rstrip())
        elif raw.lstrip().startswith("- "):
            vaciar_par(); vaciar_cita(); vaciar_tabla()
            buf_bul.append(raw.lstrip()[2:])
        else:
            # línea de continuación: pertenece a la viñeta / cita / párrafo en curso
            vaciar_tabla()
            if buf_bul:
                buf_bul[-1] += " " + s
            elif buf_cita:
                buf_cita.append(s)
            else:
                buf_par.append(s)
    vaciar_todo()

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title=os.path.basename(pdf_path))
    doc.build(flow)
    print("PDF generado:", pdf_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python md_a_pdf.py archivo.md [salida.pdf]")
        sys.exit(1)
    md = sys.argv[1]
    pdf = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(md)[0] + ".pdf"
    construir(md, pdf)
