"""Pillow wrapping and isolated Tk controls; no annotation data is touched."""
import os
import unittest

from scripts.annotate.korean_tk import render_text, install


class PillowTests(unittest.TestCase):
    def test_korean_render_wraps_without_spaces(self):
        short=render_text('가림 상태',size=19,wraplength=180)
        long=render_text('사람이확인해야하는남은코너만표시합니다',size=19,wraplength=100)
        self.assertLessEqual(long.width,106)
        self.assertGreater(long.height,short.height)


@unittest.skipUnless(os.environ.get('DISPLAY'),'requires isolated local Tk fixture')
class TkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install()

    def setUp(self):
        import tkinter as tk
        self.tk=tk
        self.root=tk.Tk();self.root.geometry('200x100+0+0');self.root.withdraw()

    def tearDown(self):
        self.root.destroy()

    def test_first_korean_configure_preserves_actual_style_and_font(self):
        self.root.option_add('*Font',('Noto Sans CJK KR',11))
        label=self.tk.Label(self.root,text='',fg='#a5e4c3',bg='#17202b')
        label.configure(text='남음 3030 · 입력 없음')
        self.assertEqual(label._korean_text_options['foreground'],'#a5e4c3')
        self.assertEqual(label._korean_text_options['background'],'#17202b')
        self.assertIn('11',str(label._korean_text_options['font']))
        self.assertTrue(label._korean_text_image.width()>0)
        label.config(text='저장 완료')
        self.assertEqual(label._korean_text_source,'저장 완료')

    def test_check_radio_selection_and_textvariable_are_real_tk_controls(self):
        value=self.tk.BooleanVar(master=self.root,value=False)
        checkbox=self.tk.Checkbutton(self.root,text='예측 노출 여부 확인',variable=value)
        checkbox.invoke();self.assertTrue(value.get())
        choice=self.tk.StringVar(master=self.root,value='a')
        radio=self.tk.Radiobutton(self.root,text='실제로 사람 판단',variable=choice,value='b')
        radio.invoke();self.assertEqual(choice.get(),'b')
        text=self.tk.StringVar(master=self.root,value='시작 전')
        label=self.tk.Label(self.root,textvariable=text)
        text.set('변경된 한글 안내')
        self.assertEqual(label._korean_text_source,'변경된 한글 안내')

    def test_messagebox_with_withdrawn_parent_is_visible_and_returns_bool(self):
        from tkinter import messagebox
        observed={}
        def accept_fixture_only():
            window=next(w for w in self.root.winfo_children() if isinstance(w,self.tk.Toplevel))
            observed['viewable']=window.winfo_viewable()
            observed['state']=window.state()
            window.event_generate('<Return>')
        self.root.after(100,accept_fixture_only)
        result=messagebox.askyesno('검증용 창','임시 검증이며 사람 주석을 만들지 않습니다.',parent=self.root)
        self.assertTrue(result)
        self.assertEqual(observed,{'viewable':1,'state':'normal'})


if __name__=='__main__':unittest.main()
