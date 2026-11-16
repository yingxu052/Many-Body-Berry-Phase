"""Build a detailed academic PDF on vertex phases and Gutzwiller projection."""

from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, Table,
    TableStyle, PageBreak, KeepTogether, Flowable,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ASSETS = OUT / "berry_summary_assets"
ASSETS.mkdir(parents=True, exist_ok=True)
PDF = OUT / "berry_phase_vertex_and_gutzwiller_summary.pdf"
PAGE_W, PAGE_H = A4
LEFT = 2.15 * cm
RIGHT = 2.15 * cm
TOP = 2.15 * cm
BOTTOM = 2.0 * cm
TEXT_W = PAGE_W - LEFT - RIGHT
NAVY = HexColor("#18324b")
BLUE = HexColor("#315d7d")
MUTED = HexColor("#5c6872")
BODY = HexColor("#26313a")
LIGHT = HexColor("#eef2f5")


# Unicode serif fonts with broad mathematical-symbol coverage.
regular = font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif"))
bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif", weight="bold"))
italic = font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif", style="italic"))
pdfmetrics.registerFont(TTFont("DVS", regular))
pdfmetrics.registerFont(TTFont("DVS-Bold", bold))
pdfmetrics.registerFont(TTFont("DVS-Italic", italic))
pdfmetrics.registerFontFamily(
    "DVS", normal="DVS", bold="DVS-Bold", italic="DVS-Italic", boldItalic="DVS-Bold"
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="BodyDVS", fontName="DVS", fontSize=10.1, leading=15.0,
    textColor=BODY, alignment=TA_LEFT, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="SmallDVS", parent=styles["BodyDVS"], fontSize=8.8, leading=12.8,
    textColor=MUTED,
))
styles.add(ParagraphStyle(
    name="TitleDVS", fontName="DVS-Bold", fontSize=24, leading=31,
    textColor=NAVY, alignment=TA_CENTER, spaceAfter=18,
))
styles.add(ParagraphStyle(
    name="SubtitleDVS", fontName="DVS", fontSize=12.2, leading=18,
    textColor=MUTED, alignment=TA_CENTER, spaceAfter=12,
))
styles.add(ParagraphStyle(
    name="H1DVS", fontName="DVS-Bold", fontSize=16, leading=21,
    textColor=NAVY, spaceBefore=13, spaceAfter=9, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="H2DVS", fontName="DVS-Bold", fontSize=12.3, leading=17,
    textColor=BLUE, spaceBefore=11, spaceAfter=6, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="H3DVS", fontName="DVS-Bold", fontSize=10.6, leading=15,
    textColor=BODY, spaceBefore=8, spaceAfter=5, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="CaptionDVS", fontName="DVS", fontSize=8.8, leading=12.5,
    textColor=MUTED, alignment=TA_CENTER, spaceBefore=4, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="CalloutDVS", parent=styles["BodyDVS"], backColor=LIGHT,
    borderColor=HexColor("#c8d4df"), borderWidth=0.6, borderPadding=9,
    spaceBefore=6, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="RefDVS", parent=styles["BodyDVS"], fontSize=9.0, leading=13.0,
    leftIndent=20, firstLineIndent=-20, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="TOCTitle", fontName="DVS-Bold", fontSize=17, leading=22,
    textColor=NAVY, spaceAfter=12,
))
styles.add(ParagraphStyle(
    name="TOC0", fontName="DVS", fontSize=9.6, leading=12,
    leftIndent=0, firstLineIndent=0, textColor=BODY,
))
styles.add(ParagraphStyle(
    name="TOC1", fontName="DVS", fontSize=8.6, leading=10.5,
    leftIndent=16, firstLineIndent=0, textColor=MUTED,
))


def para(text, style="BodyDVS"):
    return Paragraph(text, styles[style])


def heading(text, level=1, key=None):
    key = key or "sec-" + str(heading.counter)
    heading.counter += 1
    style = {1: "H1DVS", 2: "H2DVS", 3: "H3DVS"}[level]
    p = Paragraph(f'<a name="{key}"/>{text}', styles[style])
    p._toc_level = level - 1
    p._toc_text = text
    p._toc_key = key
    return p


heading.counter = 0


def equation(tex, name, max_width=TEXT_W * 0.94, fontsize=14):
    """Render one mathtext line and return a centered ReportLab image."""
    path = ASSETS / f"eq_{name}.png"
    tex = __import__("re").sub(r"\\boldsymbol\s+([A-Za-z])", r"\\boldsymbol{\1}", tex)
    tex = __import__("re").sub(r"\\mathcal\s+([A-Za-z])", r"\\mathrm{\1}", tex)
    tex = __import__("re").sub(r"\\mathbb\s+([A-Za-z])", r"\\mathrm{\1}", tex)
    tex = tex.replace(r"\mathcal{", r"\mathrm{").replace(r"\mathbb{", r"\mathrm{")
    tex = tex.replace(r"\boxed{", "{").replace(r"\frac12", r"\frac{1}{2}")
    fig = plt.figure(figsize=(0.2, 0.2))
    fig.text(0, 0, f"${tex}$", fontsize=fontsize, color="#17242e")
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.055, transparent=True)
    plt.close(fig)
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        px_w, px_h = im.size
    width = px_w / 240 * 72
    height = px_h / 240 * 72
    if width > max_width:
        scale = max_width / width
        width *= scale
        height *= scale
    img = Image(str(path), width=width, height=height)
    img.hAlign = "CENTER"
    img.spaceBefore = 3
    img.spaceAfter = 7
    return img


def eq_group(items):
    return [equation(tex, name) for name, tex in items]


def academic_table(rows, widths, font_size=8.5, header=True):
    converted = []
    for r, row in enumerate(rows):
        converted.append([
            cell if isinstance(cell, Flowable) else Paragraph(
                str(cell), ParagraphStyle(
                    f"table-{id(rows)}-{r}-{c}", parent=styles["BodyDVS"],
                    fontName="DVS-Bold" if header and r == 0 else "DVS",
                    fontSize=font_size, leading=font_size * 1.35,
                    spaceAfter=0, textColor=BODY,
                )
            )
            for c, cell in enumerate(row)
        ])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="CENTER")
    commands = [
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.append(("LINEBELOW", (0, 0), (-1, 0), 0.65, NAVY))
    table.setStyle(TableStyle(commands))
    table.spaceBefore = 5
    table.spaceAfter = 9
    return table


def make_figures():
    # Twist path.
    fig, ax = plt.subplots(figsize=(7.3, 4.7))
    pts = np.array([[0, 0], [np.pi, 0], [np.pi, np.pi], [2*np.pi, np.pi], [2*np.pi, 2*np.pi]])
    labels = ["$\\Theta_x$", "$\\Theta_y$", "$\\Theta_x$", "$T_x\\Theta_y$"]
    for i in range(4):
        a, b = pts[i], pts[i+1]
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#315d7d"))
        mid = (a+b)/2
        offset = np.array([0.0, 0.18]) if i % 2 == 0 else np.array([0.18, 0.0])
        ax.text(*(mid+offset), labels[i], ha="center", va="center", fontsize=13, color="#18324b")
    vertex_text = ["$(0,0)$", "$(\\pi,0)$", "$(\\pi,\\pi)$", "$(2\\pi,\\pi)$", "$(2\\pi,2\\pi)$"]
    offsets = [(-0.2, -0.22), (-0.15, -0.24), (-0.4, 0.18), (-0.25, -0.25), (-0.35, 0.18)]
    for p, txt, off in zip(pts, vertex_text, offsets):
        ax.scatter(*p, s=38, color="#18324b", zorder=5)
        ax.text(*(p+np.array(off)), txt, fontsize=11, color="#26313a")
    ax.set_xlim(-0.55, 2*np.pi+0.55)
    ax.set_ylim(-0.55, 2*np.pi+0.55)
    ax.set_xticks([0, np.pi, 2*np.pi], ["0", "$\\pi$", "$2\\pi$"])
    ax.set_yticks([0, np.pi, 2*np.pi], ["0", "$\\pi$", "$2\\pi$"])
    ax.set_xlabel("$\\phi_x$")
    ax.set_ylabel("$\\phi_y$")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.13)
    fig.tight_layout()
    fig.savefig(ASSETS / "twist_loop.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # TRIM sector eigenvalues.
    fig, ax = plt.subplots(figsize=(7.3, 3.7))
    positions = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    us = [1j, -1j, 1j, 1j]
    xi = [-2, 2, 2, 6]
    spinor = ["c", "d", "d", "d"]
    for (x, y), u, xx, sp in zip(positions, us, xi, spinor):
        color = "#315d7d" if u.imag > 0 else "#8a4b45"
        ax.scatter(x, y, s=150, color=color, zorder=3)
        ax.text(x, y+0.14, f"$u_k={int(u.imag)}i$", ha="center", fontsize=12, color=color)
        ax.text(x, y-0.19, f"$\\xi={xx}$, {sp}, odd", ha="center", fontsize=10, color="#26313a")
    ax.set_xticks([0, 1], ["0", "$\\pi$"])
    ax.set_yticks([0, 1], ["0", "$\\pi$"])
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-0.48, 1.48)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Fourth vertex: four self-inverse momentum sectors")
    fig.tight_layout()
    fig.savefig(ASSETS / "trim_sectors.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


make_figures()


class AcademicDoc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, "_toc_level"):
            level = flowable._toc_level
            text = flowable._toc_text
            key = flowable._toc_key
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)
            self.notify("TOCEntry", (level, text, self.page, key))


def later_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("DVS", 8.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(LEFT, PAGE_H-1.25*cm, "Many-body Berry phase: vertex representations and projection")
    canvas.drawRightString(PAGE_W-RIGHT, PAGE_H-1.25*cm, "Technical summary")
    canvas.setStrokeColor(HexColor("#b8c4ce"))
    canvas.setLineWidth(0.45)
    canvas.line(LEFT, PAGE_H-1.43*cm, PAGE_W-RIGHT, PAGE_H-1.43*cm)
    canvas.drawCentredString(PAGE_W/2, 1.1*cm, str(doc.page))
    canvas.restoreState()


def first_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(1.2)
    canvas.line(3.0*cm, PAGE_H-3.0*cm, PAGE_W-3.0*cm, PAGE_H-3.0*cm)
    canvas.line(3.0*cm, 3.0*cm, PAGE_W-3.0*cm, 3.0*cm)
    canvas.setFont("DVS", 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(PAGE_W/2, 2.25*cm, "Prepared from the model and numerical conventions used in our calculations")
    canvas.restoreState()


doc = AcademicDoc(
    str(PDF), pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
    topMargin=TOP, bottomMargin=BOTTOM,
    title="Many-Body Berry Phase on the Twist Loop",
    author="Kimi Work, prepared for Xuzhe Ying",
    subject="Real-gauge and Bloch-representation derivations, with Gutzwiller projection",
)
frame = Frame(LEFT, BOTTOM, TEXT_W, PAGE_H-TOP-BOTTOM, id="main")
doc.addPageTemplates([
    PageTemplate(id="first", frames=[frame], onPage=first_page),
    PageTemplate(id="later", frames=[frame], onPage=later_page),
])

story = []

# Cover.
story += [Spacer(1, 3.2*cm)]
story.append(para("Many-Body Berry Phase<br/>on the Twist Loop", "TitleDVS"))
story.append(para(
    "Detailed derivations using leg-specific real gauges and Bloch-spinor representations,<br/>"
    "with the Gutzwiller-projection invariance argument", "SubtitleDVS"))
story.append(Spacer(1, 1.0*cm))
story.append(para(
    "Model parameters: t = t<sub>1</sub> = 1, m = −2, Δ<sub>p</sub> = 1, Δ<sub>s</sub> = 1/2<br/>"
    "Closed trajectory: (0,0) → (π,0) → (π,π) → (2π,π) → (2π,2π)", "SubtitleDVS"))
story.append(Spacer(1, 1.2*cm))
story.append(para(
    "Main result: the vertex unitary eigenvalues are (+1,+1,+1,−1). "
    "In one continuously tracked real-gauge convention, the vertex transition functions are "
    "(1,1,−1,i), giving γ = −π/2 modulo 2π.", "CalloutDVS"))
story.append(PageBreak())
story.append(Paragraph('<a name="contents"/>Contents', styles["TOCTitle"]))
toc = TableOfContents()
toc.levelStyles = [styles["TOC0"], styles["TOC1"], styles["TOC1"]]
story.append(toc)
story.append(PageBreak())
story[-1]._next_template = "later"

# 0. Executive summary.
story.append(heading("0. Scope, conventions, and main conclusions", 1, "scope"))
story.append(para(
    "This note derives the four vertex contributions to the many-body Berry phase in two independent ways. "
    "The first uses leg-specific antiunitary symmetries and real gauges. The second uses the Bloch/Nambu "
    "representation of the unitary symmetry at each vertex and evaluates its ground-state eigenvalue momentum "
    "sector by momentum sector. The final sections extend the real-gauge argument to Gutzwiller-projected states."))
story.append(para(
    "The Wilson-loop convention and the real pairing interpolation matter. I use normalized ground-state overlaps "
    "and a real seam-pairing function s(φ), either the piecewise-linear interpolation or cos(φ). The earlier complex "
    "pairing phase exp(iφ) does not possess the same leg-specific antiunitary symmetries, so the vertex derivation "
    "below does not apply unchanged to that path."))
story += eq_group([
    ("main_wilson", r"e^{i\gamma}=\prod_s\frac{\langle\Psi(\lambda_s)|\Psi(\lambda_{s+1})\rangle}{|\langle\Psi(\lambda_s)|\Psi(\lambda_{s+1})\rangle|},\qquad \lambda_{S+1}=\lambda_1"),
    ("main_vertex", r"U_v=\Theta_{\mathrm{out}}\Theta_{\mathrm{in}},\qquad U_v|\Psi_v\rangle=u_v|\Psi_v\rangle,\qquad T_v^2=u_v"),
    ("main_values", r"(u_1,u_2,u_3,u_4)=(1,1,1,-1),\qquad (T_1,T_2,T_3,T_4)=(1,1,-1,i)"),
    ("main_gamma", r"e^{i\gamma}=T_1T_2T_3T_4=-i,\qquad \gamma=-\frac{\pi}{2}\quad(\mathrm{mod}\;2\pi)"),
])
story.append(para(
    "The individual vertex phases are gauge/sign dependent. The ground-state eigenvalues u_v, the relation "
    "T_v²=u_v, and the total Wilson loop modulo 2π are the robust statements. In particular, u_3=+1 permits "
    "both T_3=+1 and T_3=−1; continuous real-gauge tracking selects the latter in the convention used here."))

# 1. Model.
story.append(heading("1. Model, twist path, and Wilson loop", 1, "model"))
story.append(heading("1.1 Bloch/Nambu Hamiltonian", 2, "model-bloch"))
story.append(para(
    "The normal block is the two-orbital Chern-insulator Hamiltonian. The pairing block contains intraorbital "
    "p-wave terms and interorbital s-wave pairing. Only the normal block is needed in closed form for the final "
    "self-inverse momentum evaluation, but the full structure determines the antiunitary symmetries on each leg."))
story += eq_group([
    ("hci", r"H_{\mathrm{CI}}(\boldsymbol k)=\xi(\boldsymbol k)\sigma_z-2t_1\sin k_x\,\sigma_x-2t_1\sin k_y\,\sigma_y"),
    ("xi", r"\xi(\boldsymbol k)=-2t\cos k_x-2t\cos k_y-m"),
    ("hp_entries", r"H_{P,cc}=-H_{P,dd}=-\Delta_p(\sin k_x-i\sin k_y),\qquad H_{P,cd}=-H_{P,dc}=-\Delta_s"),
])
story.append(para(
    "The numerical implementation uses the corresponding real-space BdG Hamiltonian. Hopping across the seam "
    "from site 0 to site 1 acquires exp(iφ_μ). The seam-pairing terms are multiplied by the real function s(φ_μ). "
    "For the linear path, s decreases continuously from +1 to −1 on 0≤φ≤π and returns from −1 to +1 on "
    "π≤φ≤2π. For the cosine path, s(φ)=cosφ."))
story += eq_group([
    ("slinear1", r"s(\phi)=1-\frac{2\phi}{\pi}\qquad(0\leq\phi\leq\pi)"),
    ("slinear2", r"s(\phi)=\frac{2\phi}{\pi}-3\qquad(\pi\leq\phi\leq2\pi)"),
])
story.append(heading("1.2 Closed trajectory and its four real structures", 2, "model-path"))
story.append(Image(str(ASSETS / "twist_loop.png"), width=14.5*cm, height=9.35*cm))
story.append(para(
    "Figure 1. The four-leg twist loop. Each leg has its own reflection-conjugation symmetry. "
    "The last leg requires a translated y-reflection.", "CaptionDVS"))
story.append(para(
    "At every interior point of a leg the ground state is nondegenerate and gapped. The Hamiltonian closes at "
    "(2π,2π)=(0,0). The four vertices are denoted v_1=(π,0), v_2=(π,π), v_3=(2π,π), and v_4=(2π,2π)."))
story.append(heading("1.3 Gauge invariance of the discretized Wilson loop", 2, "model-wilson"))
story.append(para(
    "Under an arbitrary rephasing of each state, a normalized link changes by endpoint phases. On a closed loop "
    "these phases telescope, so the total product is invariant modulo 2π. This statement assumes that no link "
    "overlap vanishes; otherwise its phase is undefined and the mesh or gauge must be repaired."))
story += eq_group([
    ("gauge_state", r"|\widetilde\Psi_s\rangle=e^{i\beta_s}|\Psi_s\rangle"),
    ("gauge_link", r"\widetilde O_s=e^{-i\beta_s}O_s e^{i\beta_{s+1}},\qquad \prod_s\widetilde O_s=\prod_s O_s"),
])

# 2 ZQ.
story.append(heading("2. Symmetry-based Z₄ quantization", 1, "zq"))
story.append(para(
    "Before using the real gauges, it is useful to recall the more general path-symmetry argument. Suppose a "
    "closed trajectory decomposes into Q trajectories C_j, and a symmetry maps each C_j to C_{j+1} without "
    "reversing orientation. Then all loop phases equal a common γ modulo 2π. If the concatenated trajectories "
    "cancel, their phases sum to zero. These are the standard relations behind symmetry-adapted Z_Q Berry "
    "phases.<super>[1,2]</super>"))
story += eq_group([
    ("zq_equal", r"\gamma_1=\gamma_2=\cdots=\gamma_Q=\gamma\quad(\mathrm{mod}\;2\pi)"),
    ("zq_cancel", r"\sum_{j=1}^{Q}\gamma_j=Q\gamma=0\quad(\mathrm{mod}\;2\pi)"),
    ("zq_values", r"\gamma=\frac{2\pi n}{Q}\quad(\mathrm{mod}\;2\pi),\qquad Q=4:\quad \gamma\in\left\{0,\frac{\pi}{2},\pi,\frac{3\pi}{2}\right\}"),
])
story.append(para(
    "This argument constrains entire Wilson loops, not the local Berry connection at every point. A particular "
    "leg may have nonzero smooth contributions in a particular gauge. Exact quantization is a property of the "
    "symmetry-adapted closed path, not a generic property of every interpolation between the same endpoints. "
    "For an actual Hamiltonian family, the quantized sector can change only through a gap closing, loss of "
    "symmetry, or a singular state family."))

# 3 Approach I.
story.append(heading("3. Approach I: leg-specific antiunitary symmetries and real gauges", 1, "approach1"))
story.append(heading("3.1 Explicit reflection-conjugation operators", 2, "a1-ops"))
story.append(para(
    "Let K denote complex conjugation in the real-space fermion basis. Coordinates are understood modulo "
    "L_x,L_y, and Q=diag(i,−i) acts on the orbital pair (c,d). The three unitary spatial operations are:"))
story += eq_group([
    ("rx", r"R_x:(x,y)\mapsto(1-x,y),\qquad Q_x=\mathrm{diag}(1,1)"),
    ("ry", r"R_y:(x,y)\mapsto(x,1-y),\qquad Q_y=\mathrm{diag}(i,-i)"),
    ("ryt", r"R_y^{(x)}=T_xR_y:(x,y)\mapsto(x+1,1-y),\qquad Q_y^{(x)}=\mathrm{diag}(i,-i)"),
    ("theta_defs", r"\Theta_x=R_xK,\qquad \Theta_y=R_yK,\qquad \Theta_y^{(x)}=T_xR_yK"),
])
story.append(para(
    "Each operator squares to one. On the four legs, the antiunitary symmetry is respectively Θ_x, Θ_y, Θ_x, "
    "and Θ_y^(x). Complex conjugation reverses a hopping phase, while the reflection exchanges the two bond "
    "orientations and restores the original term. The orbital matrix Q supplies the signs needed by the "
    "interorbital hopping terms. Reality of s(φ) is essential."))
story += eq_group([
    ("leg_ops", r"\Theta_1=\Theta_x,\qquad \Theta_2=\Theta_y,\qquad \Theta_3=\Theta_x,\qquad \Theta_4=\Theta_y^{(x)}"),
    ("leg_sym", r"\Theta_\ell H(\lambda)\Theta_\ell^{-1}=H(\lambda),\qquad \lambda\in\mathcal C_\ell"),
])
story.append(heading("3.2 Antiunitary eigenphase and construction of a real state", 2, "a1-reality"))
story.append(para(
    "Because the ground state is unique on a leg, it is an eigenstate of that leg's antiunitary operation. The "
    "eigenvalue is a phase A_l(λ). Multiplying the state by its square root produces a state invariant under "
    "the same antiunitary operation."))
story += eq_group([
    ("a_phase", r"\Theta_\ell|\Psi(\lambda)\rangle=\mathcal A_\ell(\lambda)|\Psi(\lambda)\rangle,\qquad \mathcal A_\ell=e^{ia_\ell},\qquad |\mathcal A_\ell|=1"),
    ("real_def", r"|R_\ell\rangle=\mathcal A_\ell^{1/2}|\Psi\rangle=e^{ia_\ell/2}|\Psi\rangle"),
    ("real_proof", r"\Theta_\ell|R_\ell\rangle=e^{-ia_\ell/2}e^{ia_\ell}|\Psi\rangle=|R_\ell\rangle"),
])
story.append(heading("3.3 Why the interior of every leg has zero Berry connection", 2, "a1-interior"))
story.append(para(
    "Normalization forces the real part of the diagonal connection to vanish. Reality forces the whole diagonal "
    "matrix element to be real. Therefore it vanishes identically. Nearby overlaps within one leg are real and, "
    "after choosing continuous signs, positive."))
story += eq_group([
    ("norm_deriv", r"\partial_\lambda\langle R_\ell|R_\ell\rangle=0\quad\Longrightarrow\quad \mathrm{Re}\,\langle R_\ell|\partial_\lambda R_\ell\rangle=0"),
    ("real_conn", r"\langle R_\ell|\partial_\lambda R_\ell\rangle\in\mathbb R\quad\Longrightarrow\quad \boxed{\langle R_\ell|\partial_\lambda R_\ell\rangle=0}"),
])
story.append(para(
    "The full loop phase is consequently located at the four points where the real structure changes. This is a "
    "transition-function construction: the smooth phase inside each patch vanishes, but patches must be glued "
    "nontrivially. Real-gauge obstructions of this general type are also central to Stiefel–Whitney descriptions "
    "of real band structures.<super>[3]</super>"))
story.append(heading("3.4 Vertex transition functions and the unitary U_v", 2, "a1-unitary"))
story.append(para(
    "At a vertex, the same ground state has one real representation from the incoming leg and another from the "
    "outgoing leg. Their normalized overlap is the vertex transition function T_v."))
story += eq_group([
    ("rinout", r"|R_{\mathrm{in}}\rangle=\mathcal A_{\mathrm{in}}^{1/2}|\Psi\rangle,\qquad |R_{\mathrm{out}}\rangle=\mathcal A_{\mathrm{out}}^{1/2}|\Psi\rangle"),
    ("tv_def", r"T_v=\langle R_{\mathrm{in}}|R_{\mathrm{out}}\rangle=\left(\frac{\mathcal A_{\mathrm{out}}}{\mathcal A_{\mathrm{in}}}\right)^{1/2}"),
    ("loop_t", r"e^{i\gamma}=\prod_{v=1}^{4}T_v"),
])
story.append(para(
    "Since both antiunitary operations are symmetries at the vertex, their product is a unitary symmetry. Its "
    "ground-state eigenvalue is exactly the ratio of antiunitary amplitudes."))
story += eq_group([
    ("u_def", r"U_v=\Theta_{\mathrm{out}}\Theta_{\mathrm{in}},\qquad U_vH(v)U_v^{-1}=H(v)"),
    ("u_action1", r"U_v|\Psi\rangle=\Theta_{\mathrm{out}}(\mathcal A_{\mathrm{in}}|\Psi\rangle)=\mathcal A_{\mathrm{in}}^{*}\mathcal A_{\mathrm{out}}|\Psi\rangle"),
    ("u_eigen", r"U_v|\Psi\rangle=u_v|\Psi\rangle,\qquad u_v=\frac{\mathcal A_{\mathrm{out}}}{\mathcal A_{\mathrm{in}}}"),
    ("t_square", r"\boxed{T_v^2=u_v},\qquad \delta_v=\arg T_v=\frac12\arg u_v\quad(\mathrm{mod}\;\pi)"),
])
story.append(heading("3.5 Explicit vertex unitaries and the role of fermion parity", 2, "a1-parity"))
story.append(academic_table([
    ["Vertex", "Unitary product", "Spatial action", "Orbital action"],
    ["v₁=(π,0)", "R_yR_x", "(x,y)→(1−x,1−y)", "Q"],
    ["v₂=(π,π)", "R_xR_y", "(x,y)→(1−x,1−y)", "Q"],
    ["v₃=(2π,π)", "T_xR_yR_x", "(x,y)→(2−x,1−y)", "Q"],
    ["v₄=(2π,2π)", "R_xT_xR_y", "(x,y)→(−x,1−y)", "Q"],
], [2.7*cm, 3.4*cm, 6.0*cm, 2.0*cm], font_size=8.4))
story.append(para(
    "Every spatial map squares to the identity, while Q²=−I. Hence the single-particle unitary squares to −I. "
    "Its many-body lift squares to fermion parity, because conjugating every annihilation operator by −1 is "
    "precisely the adjoint action of fermion parity."))
story += eq_group([
    ("u_sp_square", r"\mathcal U_v^2=-I"),
    ("u_mb_square", r"\widehat U_v^2=(-1)^{\widehat N}"),
    ("u_even", r"(-1)^{\widehat N}|\Psi\rangle=+|\Psi\rangle\quad\Longrightarrow\quad u_v^2=1,\qquad u_v=\pm1"),
])
story.append(para(
    "Thus every vertex transition function is a fourth root of unity. The sign u_v itself is not fixed by the "
    "operator algebra alone: it is a quantum number of the ground state. Sections 3.6 and 4 evaluate that sign "
    "in two different representations."))
story.append(heading("3.6 Evaluating the antiunitary amplitude from a fixed Slater reference", 2, "a1-pfaffian"))
story.append(para(
    "In the numerical real-gauge approach, the Gaussian state is represented about one common Slater reference "
    "|F⟩. For modes in the reference occupation set, a particle-hole transformation d=c† is used; otherwise d=c. "
    "The state is then written in BCS form:"))
story += eq_group([
    ("bcs_ref", r"|\Psi\rangle=\mathcal N\exp\left(\frac12\sum_{ij}Z_{ij}d_i^\dagger d_j^\dagger\right)|F\rangle"),
    ("a_ratio", r"\boxed{\mathcal A_\ell=\frac{\langle F|\Theta_\ell|\Psi\rangle}{\langle F|\Psi\rangle}}"),
])
story.append(para(
    "The numerator is evaluated from the amplitude of the reflected reference configuration. If a configuration "
    "S differs from F on an even mode set I=S△F, its amplitude is a basis-ordering sign times a Pfaffian minor "
    "of Z. Reflection also supplies orbital phases and a fermionic permutation sign."))
story += eq_group([
    ("amp_pf", r"\langle S|\Psi\rangle=\mathcal N\,\epsilon(S,F)\,\mathrm{Pf}(Z_{I,I}),\qquad I=S\triangle F"),
])
story.append(para(
    "This construction is not an additional physical approximation; it is a convenient way to evaluate the "
    "antiunitary eigenphase of a Gaussian state. At the fourth vertex, for example, A_out=−A_in, giving u_4=−1. "
    "At the first three vertices, the same evaluation gives u_1=u_2=u_3=+1."))
story.append(heading("3.7 Continuous sign tracking and the third-vertex π", 2, "a1-branch"))
story.append(para(
    "The relation T_v²=u_v fixes δ_v only modulo π. A real normalized state also has a sign freedom. The branch "
    "is fixed by starting with a square root of A_l and then requiring every interior real overlap to be positive. "
    "If an interior overlap is negative, the target real state is multiplied by −1 and the process continues. At "
    "a vertex, the transition overlap is not forced to be positive; its residual phase is the vertex contribution."))
story.append(academic_table([
    ["Vertex", "u_v", "Allowed T_v", "Continuous-gauge T_v", "δ_v"],
    ["v₁", "+1", "+1 or −1", "+1", "0"],
    ["v₂", "+1", "+1 or −1", "+1", "0"],
    ["v₃", "+1", "+1 or −1", "−1", "π"],
    ["v₄", "−1", "+i or −i", "+i", "π/2"],
], [2.2*cm, 1.7*cm, 3.4*cm, 4.0*cm, 2.0*cm], font_size=8.8))
story.append(para(
    "The third vertex is the important branch subtlety: u_3=+1 does not imply δ_3=0. The continuously tracked "
    "real gauge selects T_3=−1, whose square is also +1. Flipping the sign of a real state throughout an entire "
    "leg changes both adjacent vertex phases by π; their product is unchanged, so the total Wilson loop remains "
    "invariant modulo 2π."))

# 4 Approach II.
story.append(heading("4. Approach II: Bloch-spinor and momentum-sector derivation", 1, "approach2"))
story.append(heading("4.1 Momentum representation of the vertex unitaries", 2, "a2-momentum"))
story.append(para(
    "All four spatial operations have the form S:r↦−r+r₀ and therefore map crystal momentum k to −k. I define "
    "V_v(k) through its action on creation operators. The orbital factor Q=diag(i,−i) and the reflection center "
    "r₀ determine a momentum-dependent phase."))
story += eq_group([
    ("v_creation", r"\widehat U_v\,\boldsymbol c_{\boldsymbol k}^{\dagger}\,\widehat U_v^\dagger=V_v(\boldsymbol k)\,\boldsymbol c_{-\boldsymbol k}^{\dagger}"),
    ("v_general", r"S\boldsymbol r=-\boldsymbol r+\boldsymbol r_0\quad\Longrightarrow\quad V_v(\boldsymbol k)=e^{i\boldsymbol k\cdot\boldsymbol r_0}Q"),
    ("v_square_momentum", r"V_v(-\boldsymbol k)V_v(\boldsymbol k)=-I"),
])
story.append(para(
    "For the fourth vertex, r₀=(0,1), so V_4(k)=exp(ik_y)Q. At a self-inverse momentum, k=−k modulo a reciprocal "
    "lattice vector, exp(ik_y)=±1 and V_4(k)²=−I. Its two eigenvalues are therefore ±i on single-particle states."))
story.append(heading("4.2 Which vertices contain self-inverse momenta?", 2, "a2-trim"))
story.append(para(
    "Periodic boundary conditions allow k_μ=2πn_μ/L_μ. Antiperiodic boundary conditions allow "
    "k_μ=2π(n_μ+1/2)/L_μ. A self-inverse momentum must satisfy 2k_μ=0 modulo 2π in both directions. On the even "
    "lattices used here, antiperiodic directions contain no such momentum."))
story.append(academic_table([
    ["Vertex", "x boundary", "y boundary", "Self-inverse sectors", "Result"],
    ["v₁=(π,0)", "Antiperiodic", "Periodic", "None", "u₁=+1"],
    ["v₂=(π,π)", "Antiperiodic", "Antiperiodic", "None", "u₂=+1"],
    ["v₃=(2π,π)", "Periodic", "Antiperiodic", "None", "u₃=+1"],
    ["v₄=(2π,2π)", "Periodic", "Periodic", "(0,0), (π,0), (0,π), (π,π)", "Evaluate below"],
], [2.5*cm, 2.7*cm, 2.7*cm, 5.0*cm, 2.0*cm], font_size=8.4))
story.append(para(
    "For a distinct ±k pair, the even BCS ground sector has U_v eigenvalue +1. One minus sign comes from "
    "V_v(−k)V_v(k)=−I; another comes from exchanging the two fermionic creation operators when k and −k are "
    "interchanged. Hence the first three vertices, which have no self-inverse sectors, give u_1=u_2=u_3=+1. "
    "Only v₄ requires a nontrivial ground-state representation analysis."))
story.append(heading("4.3 Normal and pairing Hamiltonian at a self-inverse momentum", 2, "a2-trim-ham"))
story.append(para(
    "At v₄, all self-inverse momenta have sin k_x=sin k_y=0. Therefore the spin-orbit terms and p-wave pairing "
    "vanish. Only ξ(k)σ_z and the interorbital Δ_s pairing remain. In the orbital basis (c_k,d_k), the normal "
    "energies are ξ and −ξ."))
story += eq_group([
    ("trim_h", r"h(\boldsymbol k)=\xi(\boldsymbol k)\sigma_z,\qquad \xi(\boldsymbol k)=-2t\cos k_x-2t\cos k_y-m"),
    ("trim_delta", r"\Delta_{cd}(\boldsymbol k)=-\Delta_s,\qquad \Delta_{dc}(\boldsymbol k)=+\Delta_s"),
])
story.append(para(
    "The local Fock space separates into odd and even parity sectors. This is a two-orbital momentum-sector "
    "Hamiltonian, not a real-space exact-enumeration calculation. The odd sector is governed directly by the "
    "two eigenvalues of h(k). The even sector consists of the empty state and the doubly occupied c_k†d_k† pair "
    "state. Since tr h(k)=0, its two energies are ±|Δ_s|."))
story += eq_group([
    ("odd_energy", r"E_{o,\pm}=\pm|\xi(\boldsymbol k)|,\qquad E_{o,\mathrm{min}}=-|\xi(\boldsymbol k)|"),
    ("even_energy", r"E_{e,\pm}=\frac12\left[\mathrm{tr}\,h\pm\sqrt{(\mathrm{tr}\,h)^2+4|\Delta_s|^2}\right]=\pm|\Delta_s|"),
    ("parity_criterion", r"E_{o,\mathrm{min}}<E_{e,\mathrm{min}}\quad\Longleftrightarrow\quad |\xi(\boldsymbol k)|>|\Delta_s|"),
])
story.append(para(
    "Thus the momentum-sector ground state is odd when |ξ|>|Δ_s| and even when |ξ|<|Δ_s|. Equality is a local "
    "gap closing. In an even sector, the empty and paired states both have V_4 eigenvalue +1. In an odd sector, "
    "the lowest Bloch spinor is c for ξ<0 and d for ξ>0, and its V_4 eigenvalue is respectively +i exp(ik_y) "
    "or −i exp(ik_y)."))
story.append(heading("4.4 Explicit evaluation at m=−2", 2, "a2-values"))
story.append(para(
    "With t=1 and m=−2, the four values ξ=−2cos k_x−2cos k_y+2 are −2, +2, +2, and +6. Their magnitudes all "
    "exceed Δ_s=1/2. Hence every self-inverse sector has an odd-parity ground state. Four odd sectors together "
    "give the required even total fermion parity."))
story.append(academic_table([
    ["k", "ξ(k)", "Lowest odd spinor", "exp(ik_y)", "Sector eigenvalue"],
    ["(0,0)", "−2", "c", "+1", "+i"],
    ["(π,0)", "+2", "d", "+1", "−i"],
    ["(0,π)", "+2", "d", "−1", "+i"],
    ["(π,π)", "+6", "d", "−1", "+i"],
], [2.2*cm, 2.0*cm, 4.0*cm, 2.7*cm, 3.4*cm], font_size=8.8))
story.append(Image(str(ASSETS / "trim_sectors.png"), width=14.4*cm, height=7.3*cm))
story.append(para(
    "Figure 2. Ground-state parity and U₄ eigenvalue in each self-inverse sector at the fourth vertex. "
    "The product of the four sector eigenvalues is −1.", "CaptionDVS"))
story += eq_group([
    ("u4_product", r"u_4=(i)(-i)(i)(i)=-1"),
    ("all_u_bloch", r"\boxed{(u_1,u_2,u_3,u_4)=(+1,+1,+1,-1)}"),
])
story.append(para(
    "This reproduces the fixed-reference Gaussian/Pfaffian calculation and the direct many-body symmetry "
    "expectation value. The derivation also explains why the result is independent of the linear system size "
    "for the even periodic lattices considered: the same four self-inverse momenta and the same local sector "
    "Hamiltonians occur for every such size."))
story.append(heading("4.5 Why all four odd sectors do not imply an odd ground state", 2, "a2-parity-check"))
story.append(para(
    "Each odd momentum sector has one occupied fermionic mode. Four such sectors therefore contain four "
    "fermions and have even total parity. Their U₄ eigenvalues are ±i rather than ±1, consistently with "
    "U₄²=(−1)^N: in an odd sector U₄²=−1, while in the full even-parity ground state U₄²=+1. The product of "
    "the four ±i values is real and equals −1."))

# 5 Combine.
story.append(heading("5. Combining the vertex data into the many-body Berry phase", 1, "combine"))
story.append(para(
    "The real-gauge calculation and the Bloch-representation calculation provide the same four squared "
    "transition functions. Choosing the continuously tracked branch described in Section 3.7 gives:"))
story += eq_group([
    ("comb_t", r"T_1=1,\qquad T_2=1,\qquad T_3=-1,\qquad T_4=i"),
    ("comb_delta", r"(\delta_1,\delta_2,\delta_3,\delta_4)=\left(0,0,\pi,\frac{\pi}{2}\right)"),
    ("comb_gamma", r"\gamma=0+0+\pi+\frac{\pi}{2}=\frac{3\pi}{2}\equiv-\frac{\pi}{2}\quad(\mathrm{mod}\;2\pi)"),
])
story.append(para(
    "A branch-independent partial check follows directly from the ground-state eigenvalues. It determines the "
    "Berry phase modulo π; the oriented Wilson-loop convention and continuous gauge determine the remaining "
    "sign modulo 2π."))
story += eq_group([
    ("comb_square", r"e^{2i\gamma}=\prod_{v=1}^{4}u_v=-1"),
    ("comb_modpi", r"\gamma=\frac{\pi}{2}\quad(\mathrm{mod}\;\pi)"),
])
story.append(heading("5.1 Numerical reality and mesh checks", 2, "combine-numerics"))
story.append(para(
    "The real-gauge Wilson product was evaluated for linear, cosine, and interpolated real pairing profiles and "
    "for M=4,8,16,32 mesh points per leg. After removing the legwise antiunitary phases, the largest interior "
    "link phase was at the 10⁻¹⁵ scale. The vertex phases remained (0,0,π,π/2), and their sum reproduced the "
    "original Berry phase to floating-point precision."))
story.append(academic_table([
    ["Quantity", "Representative numerical result"],
    ["Maximum interior stripped phase (unprojected scans)", "approximately 5×10⁻¹⁵ or smaller"],
    ["Vertex phases", "(0,0,π,π/2)"],
    ["Total Berry phase", "−1.5707963267949 radians"],
    ["Modular closure error", "approximately 2.45×10⁻¹⁶"],
    ["Projected M=4 maximum interior stripped phase", "2.63×10⁻¹⁵"],
    ["Projected total Berry phase", "−1.5707963267949 radians"],
], [8.4*cm, 6.4*cm], font_size=8.8))

# 6 Projection.
story.append(heading("6. Gutzwiller-projected states", 1, "projection"))
story.append(heading("6.1 Projected overlaps and Wilson loop", 2, "projection-overlap"))
story.append(para(
    "Let P_g be Hermitian and independent of the twist parameter λ. It may be a genuine projector or a smooth "
    "interpolation such as exp(−gD). Define the unnormalized projected state, its norm, and the normalized "
    "projected state as follows."))
story += eq_group([
    ("proj_chi", r"|\chi_g(\lambda)\rangle=P_g|\Psi(\lambda)\rangle"),
    ("proj_norm", r"N_g(\lambda)=\langle\chi_g|\chi_g\rangle=\langle\Psi(\lambda)|P_g^2|\Psi(\lambda)\rangle"),
    ("proj_state", r"|\Psi_g(\lambda)\rangle=N_g(\lambda)^{-1/2}P_g|\Psi(\lambda)\rangle"),
    ("proj_overlap", r"O_g(\lambda,\lambda')=\frac{\langle\Psi(\lambda)|P_g^2|\Psi(\lambda')\rangle}{\sqrt{N_g(\lambda)N_g(\lambda')}}"),
    ("proj_wilson", r"e^{i\gamma_g}=\prod_s\frac{O_g(\lambda_s,\lambda_{s+1})}{|O_g(\lambda_s,\lambda_{s+1})|}"),
])
story.append(para(
    "For a genuine projector P_g²=P_g, whereas for P_g=exp(−gD) the overlap contains exp(−2gD). There is no "
    "generic reason for γ_g to equal the unprojected phase. Projection changes configuration amplitudes and "
    "therefore changes both magnitudes and phases of link overlaps. Projected-parton Berry phases constructed "
    "from such overlaps have previously been evaluated by Monte Carlo sampling.<super>[4]</super>"))
story.append(heading("6.2 Sufficient condition for exact invariance under projection", 2, "projection-invariance"))
story.append(para(
    "The strong condition is that P_g respects every leg-specific antiunitary operation, not merely that it is "
    "real or diagonal in one fixed occupation basis."))
story += eq_group([
    ("proj_commute", r"\boxed{\Theta_\ell P_g\Theta_\ell^{-1}=P_g\qquad\mathrm{for\;every\;leg\;}\ell}"),
    ("proj_phase1", r"\Theta_\ell P_g|\Psi\rangle=P_g\Theta_\ell|\Psi\rangle=\mathcal A_\ell P_g|\Psi\rangle"),
])
story.append(para(
    "The projected norm is real, so the normalized projected state has exactly the same antiunitary eigenphase "
    "A_l. Multiplying by A_l^(1/2) therefore makes it real on the same leg. Interior Berry connections vanish "
    "just as in the unprojected case. It remains to show that the vertex transition functions are also unchanged."))
story += eq_group([
    ("proj_real", r"|R_{g,\ell}\rangle=\mathcal A_\ell^{1/2}|\Psi_g\rangle,\qquad \Theta_\ell|R_{g,\ell}\rangle=|R_{g,\ell}\rangle"),
    ("proj_vertex1", r"T_v^{(g)}=\langle R_{g,\mathrm{in}}|R_{g,\mathrm{out}}\rangle=\left(\frac{\mathcal A_{\mathrm{out}}}{\mathcal A_{\mathrm{in}}}\right)^{1/2}\frac{\langle\Psi|P_g^2|\Psi\rangle}{N_g}"),
    ("proj_vertex2", r"N_g=\langle\Psi|P_g^2|\Psi\rangle\quad\Longrightarrow\quad T_v^{(g)}=\left(\frac{\mathcal A_{\mathrm{out}}}{\mathcal A_{\mathrm{in}}}\right)^{1/2}=T_v^{(0)}"),
    ("proj_same", r"\boxed{\gamma_g=\gamma_0\quad(\mathrm{mod}\;2\pi)}"),
])
story.append(para(
    "At a fixed vertex, the numerator is precisely the projected norm, so all projection dependence cancels. "
    "This proves equality vertex by vertex, without relying only on continuity in g. The proof requires nonzero "
    "projected norms, smooth states on each leg, and nonzero link overlaps. It does not construct a projected "
    "parent Hamiltonian and therefore does not by itself prove that the projected state is gapped."))
story.append(heading("6.3 Commutation with C₄ alone: a weaker statement", 2, "projection-c4"))
story.append(para(
    "If instead C₄P_gC₄⁻¹=P_g, the relations among the four projected Wilson loops survive, and trajectory "
    "cancellation is independent of projection. Hence Z₄ quantization survives. At one isolated projected point, "
    "this symmetry alone does not explicitly identify the sector. If a smooth nonsingular interpolation in g is "
    "also available, continuity together with quantization prevents the sector from jumping along that family. "
    "The stronger antiunitary commutation condition proves the same transition functions directly."))
story.append(heading("6.4 Why parton-type and double-occupancy projections can differ", 2, "projection-difference"))
story.append(para(
    "The distinction is not simply correlated versus uncorrelated, or diagonal versus off-diagonal. It is whether "
    "the complete symmetry structure of the path is preserved. The parton-type projection used in our calculation "
    "preserved the Berry phase, consistently with preservation of the leg-specific real structures. A "
    "double-occupancy penalty need not do so."))
story += eq_group([
    ("double_def", r"P_g=e^{-g\widehat D},\qquad \widehat D=\sum_i n_{i\uparrow}n_{i\downarrow}"),
    ("double_theta", r"\Theta_\ell=U_\ell K\quad\Longrightarrow\quad \Theta_\ell e^{-g\widehat D}\Theta_\ell^{-1}=e^{-gU_\ell\widehat D U_\ell^\dagger}"),
])
story.append(para(
    "The last expression equals exp(−gD) only if U_lD U_l†=D. When U_l changes the local density or pairing "
    "structure, the condition can fail. A real diagonal factor leaves configuration phases unchanged but can "
    "still destroy the leg-specific real gauge by changing amplitudes. Interior overlaps then acquire imaginary "
    "parts and a smooth nonquantized contribution is allowed. This is the general diagnostic consistent with our "
    "two numerical behaviors; the commutators must be checked for each concrete implementation."))
story.append(academic_table([
    ["Condition on P_g", "Consequence"],
    ["Commutes with every leg-specific Θ_l", "Same real-gauge vertices and exactly the same total Berry phase"],
    ["Commutes with the C₄ path symmetry only", "Z₄ quantization survives; sector identification needs an additional argument"],
    ["Does not preserve the required symmetries", "A smooth nonquantized contribution is generically allowed"],
    ["Zero projected norm or singular link overlaps", "The Wilson-loop argument breaks down"],
], [7.1*cm, 7.7*cm], font_size=8.6))
story.append(heading("6.5 Relation to Hetényi’s exact finite-g example", 2, "projection-hetenyi"))
story.append(para(
    "Hetényi studied a different geometric response, namely the current response of projected variational "
    "wavefunctions. For a Gutzwiller-projected Fermi sea, the total-position shift commutes with the double-"
    "occupancy operator, so the same geometric response is obtained at finite projection strength. This is a "
    "special example of the same commutation principle, not a universal theorem for all projections and twist "
    "paths. Non-centrosymmetric or singular projectors can fall outside the argument.<super>[5]</super>"))
story += eq_group([
    ("hetenyi", r"\widehat U(\Delta X)e^{-g\widehat D}|FS\rangle=e^{i\Delta X\sum_i k_i}e^{-g\widehat D}|FS\rangle"),
])

# 7 caveats.
story.append(heading("7. Assumptions, failure modes, and physical interpretation", 1, "caveats"))
story.append(heading("7.1 Checklist for applying the vertex derivation", 2, "checklist"))
story.append(para(
    "1. The ground state must be unique and gapped on every leg and at every vertex.<br/>"
    "2. The seam-pairing function must be real for the stated reflection-conjugation symmetries.<br/>"
    "3. The correct orbital phases and the translation on the last leg must be included.<br/>"
    "4. Fermion parity is essential when converting U_v²=−I into the allowed many-body eigenvalues.<br/>"
    "5. The square root of A_l and the signs of real states must be tracked continuously.<br/>"
    "6. No normalized link overlap may vanish.<br/>"
    "7. Only the total Berry phase modulo 2π is independent of the legwise sign conventions."))
story.append(heading("7.2 What the Berry phase does not prove", 2, "interpretation"))
story.append(para(
    "Equal Berry phases do not imply equal correlation functions, Gaussian character, excitation spectra, "
    "adiabatic connectedness, or identical topological order. A projected wavefunction can be strongly "
    "non-Gaussian while retaining exactly the same Wilson loop. Likewise, the symmetry argument for a "
    "wavefunction family does not by itself establish a gapped parent Hamiltonian or identify the full vortex "
    "response of a physical spin model. The projected-parton literature also emphasizes the role of the "
    "background internal gauge flux in obtaining the correct physical response.<super>[4]</super>"))
story.append(heading("7.3 Relation to a single fixed antiunitary symmetry", 2, "single-antiunitary"))
story.append(para(
    "If one fixed antiunitary symmetry applied along the entire loop, a single real gauge could be used "
    "everywhere and the total phase would be restricted to 0 or π.<super>[6]</super> Our loop instead uses "
    "different real structures on different legs. The transition functions between those structures permit "
    "±π/2 contributions and are the reason the Z₄ result is compatible with legwise reality."))

# Appendix A.
story.append(heading("Appendix A. Detailed momentum-sector algebra", 1, "appendix-a"))
story.append(heading("A.1 Deriving the phase exp(ik·r₀)", 2, "appendix-phase"))
story.append(para(
    "Use c_k=N_cell^(−1/2) Σ_r exp(−ik·r)c_r and S²=1 with S r=−r+r₀. Acting on a creation operator and "
    "substituting r=S r′ gives exp(ik·r)=exp(ik·r₀)exp(−ik·r′). The remaining sum is c_(−k)†. This proves "
    "V(k)=exp(ik·r₀)Q. The relation V(−k)V(k)=−I follows either from Q²=−I or directly from U²=−I on "
    "single-particle states."))
story.append(heading("A.2 Even-sector energies at a self-inverse momentum", 2, "appendix-even"))
story.append(para(
    "At self-inverse k, let |P⟩=c_k†d_k†|0⟩. The even subspace is spanned by |0⟩ and |P⟩. The normal "
    "Hamiltonian contributes zero to |0⟩ and tr h(k) to |P⟩. The interorbital pairing couples these two states "
    "with magnitude |Δ_s|. Thus the even Hamiltonian has trace tr h and determinant −|Δ_s|², yielding the "
    "eigenvalues displayed in Section 4.3. The odd subspace contains c_k†|0⟩ and d_k†|0⟩ and is governed "
    "directly by h(k). Comparing the lowest energies of the two sectors determines the ground parity without "
    "constructing the full real-space Fock space."))
story.append(heading("A.3 Sector symmetry eigenvalues", 2, "appendix-sector-u"))
story.append(para(
    "At v₄, V_4(k)=exp(ik_y)diag(i,−i). An even empty state has eigenvalue +1. The paired state acquires the "
    "product of the c and d creation phases, namely exp(2ik_y)(i)(−i)=+1. An odd c-occupied state has "
    "eigenvalue i exp(ik_y), while an odd d-occupied state has eigenvalue −i exp(ik_y). These four rules are "
    "sufficient to reproduce every entry in the v₄ table."))
story += eq_group([
    ("app_c", r"u_c(\boldsymbol k)=i e^{ik_y}"),
    ("app_d", r"u_d(\boldsymbol k)=-i e^{ik_y}"),
    ("app_pair", r"u_{cd}(\boldsymbol k)=e^{2ik_y}(i)(-i)=+1"),
])

# Appendix B reproducibility.
story.append(heading("Appendix B. Numerical cross-checks and reproducibility", 1, "appendix-b"))
story.append(para(
    "The following independent calculations were compared in the workspace:"))
story.append(academic_table([
    ["Calculation", "File", "Checked result"],
    ["Real-gauge vertex decomposition", "gaussian_checks/vertex_real_gauge_berry.py", "Interior phases vanish; vertex sum equals Wilson loop"],
    ["Vertex unitary eigenvalue relation", "gaussian_checks/verify_vertex_phase_algebra.py", "u=exp(2iδ) at all four vertices"],
    ["Exact small-Fock-space symmetry evaluation", "gaussian_checks/exact_vertex_eigenvalues_2x2.py", "u=(+1,+1,+1,−1) at m=−2"],
    ["Bloch-sector representation", "gaussian_checks/bloch_vertex_representation.py", "Sector eigenvalues i,−i,i,i at v₄"],
    ["Projected real-gauge calculation", "gaussian_checks/vertex_real_gauge_berry.py", "Same total phase after one-parton-per-cell projection"],
], [4.3*cm, 6.8*cm, 4.2*cm], font_size=7.7))
story.append(para(
    "The Bloch-sector script uses only the normal Bloch spinors at self-inverse momenta, the Δ_s even-sector "
    "energy, and V₄(k). Its output was compared programmatically with the many-body result; both give "
    "(+1,+1,+1,−1), with product −1."))

# References.
story.append(heading("References", 1, "references"))
refs = [
    ("[1] Kariyado, T., Morimoto, T., &amp; Hatsugai, Y. (2018). <i>Z<sub>N</sub> Berry phases in symmetry protected topological phases</i>. Physical Review Letters, 120, 247202. ", "https://arxiv.org/abs/1709.01546"),
    ("[2] Araki, H., Mizoguchi, T., &amp; Hatsugai, Y. (2020). <i>Z<sub>Q</sub> Berry phase for higher-order symmetry-protected topological phases</i>. Physical Review Research, 2, 012009(R). ", "https://arxiv.org/abs/1906.00218"),
    ("[3] Ahn, J., Park, S., Kim, D., Kim, Y., &amp; Yang, B.-J. (2019). <i>Stiefel–Whitney classes and topological phases in band theory</i>. arXiv:1904.00336. ", "https://arxiv.org/abs/1904.00336"),
    ("[4] Liu, Z.-X., Mei, J.-W., Ye, P., &amp; Wen, X.-G. (2014). <i>U(1)×U(1) symmetry-protected topological order in Gutzwiller wave functions</i>. Physical Review B, 90, 235146. ", "https://arxiv.org/abs/1408.1676"),
    ("[5] Hetényi, B. (2012). <i>Current response in extended systems as a geometric phase: Application to variational wavefunctions</i>. Journal of the Physical Society of Japan, 81, 124711. ", "https://arxiv.org/abs/1208.0671"),
    ("[6] Hatsugai, Y. (2006). <i>Quantized Berry phases as a local order parameter of a quantum liquid</i>. Journal of the Physical Society of Japan, 75, 123601. ", "https://doi.org/10.1143/JPSJ.75.123601"),
]
for text, url in refs:
    story.append(Paragraph(text + f'<a href="{url}" color="#315d7d">{url}</a>', styles["RefDVS"]))

# Ensure the second page onward uses the later-page template.
story.insert(2, __import__("reportlab.platypus", fromlist=["NextPageTemplate"]).NextPageTemplate("later"))
doc.multiBuild(story)
print(f"Saved {PDF}")
