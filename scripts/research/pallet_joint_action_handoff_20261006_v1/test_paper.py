"""Publication integrity failures that must block a success receipt."""
import tempfile
import unittest
from pathlib import Path

from scripts.research.pallet_joint_action_handoff_20261006_v1 import paper


class PaperIntegrityTests(unittest.TestCase):
    def test_explicit_missing_pdf_is_not_satisfied_by_png(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / paper.DOC / 'paper_updated'
            (target / 'audit').mkdir(parents=True)
            (target / 'main.tex').write_text(
                r'\begin{document}\cite{a}\includegraphics{plot.pdf}\end{document}')
            (target / 'supplement.tex').write_text(r'\begin{document}\end{document}')
            (target / 'plot.png').write_bytes(b'fixture-only')
            (target / 'references.bib').write_text('@article{a,\n title = {A},\n year = {2020}\n}\n')
            result = paper.validate(root)
            self.assertFalse(result['passed'])
            self.assertEqual(result['documents']['main.tex']['missing_graphics'], ['plot.pdf'])

    def test_duplicate_reference_key_is_rejected(self):
        entry = '@article{a,\n title = {A},\n year = {2020}\n}\n'
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            paper.bib_entries(entry + entry)

    def test_citation_order_uses_only_actual_includes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'main.tex').write_text('% \\cite{fake}\n\\cite{a}\\input{section}\\cite{c}')
            (root / 'section.tex').write_text(r'\cite{b,a}')
            (root / 'unused.tex').write_text(r'\cite{fake}')
            _, uses = paper.expand(root, 'main.tex')
            self.assertEqual([u['key'] for u in uses], ['a', 'b', 'a', 'c'])

    def test_style_or_graphic_change_invalidates_build_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'main.tex').write_text(
                r'\documentclass{local}\usepackage{localstyle}\includegraphics{plot.png}')
            (root / 'supplement.tex').write_text(r'\begin{document}\end{document}')
            (root / 'local.cls').write_text('% class fixture\n')
            style = root / 'localstyle.sty'
            style.write_text('% first style\n')
            graphic = root / 'plot.png'
            graphic.write_bytes(b'first graphic fixture')
            receipt = {'built_source_sha256': paper.build_bindings(root)}
            self.assertTrue(paper.build_receipt_current(root, receipt))
            style.write_text('% changed style\n')
            self.assertFalse(paper.build_receipt_current(root, receipt))
            receipt = {'built_source_sha256': paper.build_bindings(root)}
            graphic.write_bytes(b'changed graphic fixture')
            self.assertFalse(paper.build_receipt_current(root, receipt))

    def test_generated_figure_has_source_binding_before_and_pdf_after_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'figures').mkdir()
            (root / 'main.tex').write_text(r'\includegraphics{figures/plot.pdf}')
            (root / 'supplement.tex').write_text(r'\begin{document}\end{document}')
            source = root / 'figures/plot.tex'
            source.write_text(r'\documentclass{standalone}\begin{document}fixture\end{document}')
            before = paper.build_bindings(root, include_generated=False)
            self.assertIn('figures/plot.tex', before)
            (root / 'figures/plot.pdf').write_bytes(b'generated fixture')
            self.assertEqual(before, paper.build_bindings(root, include_generated=False))
            self.assertIn('figures/plot.pdf', paper.build_bindings(root))
            self.assertTrue(paper.is_build_diagnostic(root / 'main.aux'))
            self.assertTrue(paper.is_build_diagnostic(root / 'main.bbl'))
            self.assertFalse(paper.is_build_diagnostic(root / 'main.pdf'))


if __name__ == '__main__':
    unittest.main()
