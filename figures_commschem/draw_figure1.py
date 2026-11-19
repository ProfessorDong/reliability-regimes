"""Render Figure 1 as a vector PDF from its selection record. No model fitting."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath
from fontTools.pens.basePen import BasePen
from fontTools.svgLib.path import parse_path
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D


HERE = Path(__file__).resolve().parent
INK = '#192b38'
MUTED = '#405766'
BLUE = '#195983'
TEAL = '#166956'
RULE = '#bfcbd0'


class PlotPen(BasePen):
    """Translate the bond and atom-label paths made by RDKit to matplotlib paths."""
    def __init__(self):
        super().__init__(None)
        self.vertices = []
        self.codes = []
        self.start = None

    def _moveTo(self, p):
        self.start = p
        self.vertices.append(p)
        self.codes.append(PlotPath.MOVETO)

    def _lineTo(self, p):
        self.vertices.append(p)
        self.codes.append(PlotPath.LINETO)

    def _curveToOne(self, a, b, c):
        self.vertices.extend([a, b, c])
        self.codes.extend([PlotPath.CURVE4] * 3)

    def _qCurveToOne(self, a, b):
        self.vertices.extend([a, b])
        self.codes.extend([PlotPath.CURVE3] * 2)

    def _closePath(self):
        self.vertices.append(self.start)
        self.codes.append(PlotPath.CLOSEPOLY)

    def _endPath(self):
        pass


def draw_molecule(fig, mol, rect):
    left, top, width, height = rect
    axes = fig.add_axes((left / W, 1 - (top + height) / H, width / W, height / H))
    axes.set_xlim(0, width)
    axes.set_ylim(height, 0)
    axes.axis('off')
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.035
    options.bondLineWidth = 1.55
    options.minFontSize = 11
    options.setAtomPalette({6: (0.11, 0.17, 0.20), 1: (0.11, 0.17, 0.20),
                            7: (0.08, 0.23, 0.67), 8: (0.76, 0.12, 0.12),
                            9: (0.03, 0.43, 0.32), 17: (0.03, 0.43, 0.32)})
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    for elem in ET.fromstring(drawer.GetDrawingText()):
        if elem.tag.rsplit('}', 1)[-1] != 'path':
            raise ValueError(f'Unexpected molecule shape: {elem.tag}')
        pen = PlotPen()
        parse_path(elem.attrib['d'], pen)
        style = dict(elem.attrib)
        style.update(dict(item.split(':', 1) for item in style.get('style', '').split(';') if item))
        width_in_points = float(style.get('stroke-width', '0').removesuffix('px')) * POINTS_PER_UNIT
        patch = PathPatch(PlotPath(pen.vertices, pen.codes),
                          facecolor=style.get('fill', '#000000'),
                          edgecolor=style.get('stroke', 'none'), linewidth=width_in_points,
                          capstyle=style.get('stroke-linecap', 'butt'),
                          joinstyle=style.get('stroke-linejoin', 'miter'))
        if 'stroke-dasharray' in style:
            dashes = [float(v) * POINTS_PER_UNIT for v in style['stroke-dasharray'].split(',')]
            patch.set_linestyle((0, dashes))
        axes.add_patch(patch)


def aligned_molecules(example):
    query = Chem.MolFromSmiles(example['query']['smiles'])
    neighbor = Chem.MolFromSmiles(example['nearest_training']['smiles'])
    # Lay out the larger molecule first so an extra substituent cannot collide
    # with its fixed core when the shorter structure is aligned to it.
    reference, aligned = sorted([query, neighbor], key=lambda m: -m.GetNumAtoms())
    rdDepictor.Compute2DCoords(reference, nFlipsPerSample=3, nSample=180,
                               sampleSeed=31, permuteDeg4Nodes=True)
    mcs = rdFMCS.FindMCS([query, neighbor], ringMatchesRingOnly=True, completeRingsOnly=True,
                         bondCompare=rdFMCS.BondCompare.CompareOrderExact, timeout=15)
    if mcs.canceled:
        raise RuntimeError('Maximum common substructure alignment timed out.')
    core = Chem.MolFromSmarts(mcs.smartsString)
    a = reference.GetSubstructMatch(core)
    b = aligned.GetSubstructMatch(core)
    rdDepictor.GenerateDepictionMatching2DStructure(aligned, reference, list(zip(a, b)))
    # Rotate both structures in the plane to fit the landscape drawing windows.
    xy = reference.GetConformer().GetPositions()[:, :2]
    angles = np.arange(-90, 90) * np.pi / 180
    angle = min(angles, key=lambda a: np.ptp(xy @ [np.sin(a), np.cos(a)]))
    for mol in (query, neighbor):
        conf = mol.GetConformer()
        for i, (x, y, z) in enumerate(conf.GetPositions()):
            conf.SetAtomPosition(i, (x*np.cos(angle) - y*np.sin(angle),
                                     x*np.sin(angle) + y*np.cos(angle), z))
    return query, neighbor


W = 920
H = 772
FIG_WIDTH_INCH = 7.05
POINTS_PER_UNIT = FIG_WIDTH_INCH * 72 / W


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'figure1_examples.json')
    parser.add_argument('--out', type=Path, default=HERE / 'figure1_structures.pdf')
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    a, b = data['examples']
    assert a['nearest_lead'] == b['nearest_lead']
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(FIG_WIDTH_INCH, FIG_WIDTH_INCH * H / W), facecolor='white')

    def label(x, y, value, size=8, color=INK, weight='normal', align='left'):
        fig.text(x/W, 1-y/H, value, size=size, color=color, weight=weight,
                 ha=align, va='center')

    def rule(x1, x2, y, color=RULE, linewidth=0.48):
        fig.add_artist(Line2D([x1/W, x2/W], [1-y/H, 1-y/H], transform=fig.transFigure,
                              color=color, linewidth=linewidth))

    def panel(x, y, letter, title):
        label(x, y, letter, 10, INK, 'bold')
        label(x + 26, y, title, 9, INK, 'bold')

    panel(10, 20, 'a', 'Common starting lead')
    label(36, 64, '10 active leads drawn from 2,361 training compounds', 7.0, MUTED)
    label(36, 94, 'Tanimoto similarity to A', 7.7)
    label(286, 94, f"{a['lead_similarity']:.3f}", 8.2, BLUE, 'bold')
    label(36, 123, 'Tanimoto similarity to B', 7.7)
    label(286, 123, f"{b['lead_similarity']:.3f}", 8.2, BLUE, 'bold')
    lead = Chem.MolFromSmiles(a['nearest_lead']['smiles'])
    rdDepictor.Compute2DCoords(lead)
    draw_molecule(fig, lead, (456, 5, 449, 150))
    rule(10, 910, 162)

    for row, example in enumerate([a, b]):
        letter = 'AB'[row]
        top = 177 + 237 * row
        panel(10, top + 6, 'bc'[row], f'Held-out compound {letter}')
        label(480, top + 6, f'Closest training neighbor of {letter}', 9, INK, 'bold')
        query, neighbor = aligned_molecules(example)
        draw_molecule(fig, query, (12, top + 25, 430, 153))
        draw_molecule(fig, neighbor, (475, top + 25, 431, 153))
        label(36, top + 194, f"Measured  {example['query']['measured_pActivity']:.2f}", 8.1)
        label(233, top + 194, f"Predicted  {example['predicted_pActivity']:.2f}", 8.1)
        label(36, top + 215, 'pActivity = −log₁₀(C / 1 M)', 7.0, MUTED)
        label(480, top + 194, 'Tanimoto similarity', 8.1)
        label(692, top + 194, f"{example['training_similarity']:.3f}", 8.4, TEAL, 'bold')
        rule(10, 910, top + 231)

    panel(10, 678, 'd', 'Prediction error in the third of NK1R compounds farthest from the starting leads')
    cohort = data['original_cohort_summary']
    for name, x in [('nearer', 36), ('farther', 480)]:
        count = cohort[f'{name}_n']
        value = cohort[f'{name}_rmse']
        label(x, 711, f'{name.capitalize()} to training compounds (n = {count})', 7.5, MUTED)
        label(x, 742, f'RMSE  {value:.2f} pActivity', 8.6, INK, 'bold')
    fig.savefig(args.out, metadata={'Title': 'Two chemical reference sets for NK1R predictions',
                                   'CreationDate': None, 'ModDate': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
