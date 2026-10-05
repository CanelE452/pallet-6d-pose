"""Readable Korean Tk controls when the local Tk build lacks Unicode fonts.

Call install() before building a Tk UI. Importing/installing creates no window.
Only process-local widget methods are adapted; Tk still owns buttons, checkbox
indicators, radio selection, keyboard focus and selection variables.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont

FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
_INSTALLED = False


@lru_cache(maxsize=64)
def _font(size, bold=False):
    path = FONT.with_name('NotoSansCJK-Bold.ttc') if bold else FONT
    if not path.exists(): path = FONT
    return ImageFont.truetype(str(path), max(12, int(size)))


def _font_size(value, master=None):
    size, bold = 14, False
    if isinstance(value, (tuple, list)):
        if len(value)>1:
            try: size = abs(int(value[1]))
            except (ValueError, TypeError): pass
        bold = any('bold' in str(item).lower() for item in value)
    elif hasattr(value, 'actual'):
        actual=value.actual();size=abs(int(actual.get('size',14)))
        bold=actual.get('weight')=='bold'
    elif isinstance(value, str):
        match=re.search(r'\s(-?\d+)(?:\s|$)',value)
        if match:size=abs(int(match.group(1)))
        bold='bold' in value.lower()
        if master is not None:
            try:
                from tkinter import font as tkfont
                actual=tkfont.Font(root=master,font=value).actual()
                size=abs(int(actual.get('size',size)));bold=actual.get('weight')=='bold'
            except Exception:pass
    return round(size*4/3), bold


def _rgb(value, master=None, fallback='white'):
    if master is not None:
        try:
            values=master.winfo_rgb(value)
            return tuple(round(v/257) for v in values)
        except Exception: pass
    try:
        from PIL import ImageColor
        return ImageColor.getrgb(value)
    except (ValueError, TypeError):
        from PIL import ImageColor
        return ImageColor.getrgb(fallback)


def render_text(text, *, size=19, bold=False, foreground='#202020', background='#d9d9d9',
                wraplength=680, justify='left', master=None):
    """Return a Pillow image; line wrapping also works without spaces in Korean."""
    font=_font(size,bold);probe=ImageDraw.Draw(Image.new('RGB',(1,1)))
    limit=max(30,int(wraplength or 680));lines=[]
    for paragraph in str(text).split('\n'):
        current=''
        for character in paragraph:
            trial=current+character
            if current and probe.textlength(trial,font=font)>limit:
                lines.append(current.rstrip());current=character.lstrip()
            else:current=trial
        lines.append(current)
    widths=[int(probe.textlength(line,font=font)+.999) for line in lines]
    lineheight=max(size+7,font.getbbox('가Ag')[3]+6)
    width=max(1,max(widths,default=1)+6);height=max(1,len(lines)*lineheight+4)
    image=Image.new('RGB',(width,height),_rgb(background,master,'#d9d9d9'))
    draw=ImageDraw.Draw(image);color=_rgb(foreground,master,'#202020')
    for i,line in enumerate(lines):
        x=3+(width-6-widths[i])//2 if justify=='center' else width-3-widths[i] if justify=='right' else 3
        draw.text((x,2+i*lineheight),line,font=font,fill=color)
    return image


def _to_image(widget, text, options):
    from PIL import ImageTk
    size,bold=_font_size(options.get('font'),widget.master)
    background=options.get('background',options.get('bg'))
    if background is None:
        try:background=widget.master.cget('background')
        except Exception:background='#d9d9d9'
    foreground=options.get('foreground',options.get('fg'))
    if foreground is None:
        rgb=_rgb(background,widget.master,'#d9d9d9')
        foreground='#eeeeee' if sum(rgb)/3<128 else '#202020'
    image=render_text(text,size=size,bold=bold,foreground=foreground,background=background,
        wraplength=options.get('wraplength',680),justify=options.get('justify','left'),master=widget.master)
    photo=ImageTk.PhotoImage(image,master=widget.master)
    widget._korean_text_image=photo
    return photo


def install():
    """Patch four tk widgets and common messageboxes once in this process."""
    global _INSTALLED
    if _INSTALLED:return
    if not FONT.exists():raise FileNotFoundError('Korean text font missing: '+str(FONT))
    import tkinter as tk
    from tkinter import messagebox

    def merge(cnf,kw):
        values=dict(tk._cnfmerge(cnf) or {}) if cnf is not None else {}
        values.update(kw);return values

    def needs_image(widget,options):
        return bool(getattr(widget,'_korean_text_active',False) or
            any(ord(char)>127 for char in str(options.get('text',''))))

    def adapt(widget,options):
        options=dict(options)
        for short,long in [('bg','background'),('fg','foreground')]:
            if short in options:options[long]=options.pop(short)
        # Keep selection variables; only the textvariable is translated to an
        # image update callback. No reviewer selections are made here.
        variable=options.pop('textvariable',None)
        if variable is not None:
            if not isinstance(variable,tk.Variable):variable=tk.StringVar(master=widget.master,name=str(variable))
            widget._korean_text_variable=variable
            options['text']=variable.get()
        defaults={}
        if getattr(widget,'_w',None):
            # A label may start with empty ASCII text and acquire Korean through
            # configure(). Its real colour/font must survive that first update.
            for field in ('font','foreground','background','wraplength','justify'):
                try:defaults[field]=widget.cget(field)
                except tk.TclError:pass
        else:
            try:
                font_option=widget.master.option_get('font','Font')
                if font_option:defaults['font']=font_option
            except tk.TclError:pass
        old=getattr(widget,'_korean_text_options',{})
        combined={**defaults,**old,**options}
        if needs_image(widget,combined):
            widget._korean_text_active=True
            widget._korean_text_source=str(combined.get('text',''))
            widget._korean_text_options=combined
            options['image']=_to_image(widget,widget._korean_text_source,combined)
            options['text']=''
            # Tk width/height become pixels for image labels; character widths
            # from older UIs would clip the generated Korean bitmap.
            options.pop('width',None);options.pop('height',None)
            if combined.get('width') and not combined.get('wraplength'):
                combined['wraplength']=max(100,int(combined['width'])*12)
                options['image']=_to_image(widget,widget._korean_text_source,combined)
        elif variable is not None:
            options['textvariable']=variable
        return options,variable

    for cls in (tk.Label,tk.Button,tk.Checkbutton,tk.Radiobutton):
        original_init=cls.__init__;original_configure=cls.configure

        def constructor(self,master=None,cnf=None,_original=original_init,**kw):
            self.master=master or tk._get_default_root()
            values=merge(cnf,kw);values,variable=adapt(self,values)
            _original(self,self.master,values)
            if variable is not None:
                def update(*_):
                    try:self.configure(text=variable.get())
                    except tk.TclError:pass
                self._korean_text_trace=variable.trace_add('write',update)
                def cleanup(_):
                    try:variable.trace_remove('write',self._korean_text_trace)
                    except tk.TclError:pass
                self.bind('<Destroy>',cleanup,add='+')

        def configure(self,cnf=None,_original=original_configure,**kw):
            if isinstance(cnf,str) or (cnf is None and not kw):return _original(self,cnf,**kw)
            values=merge(cnf,kw)
            # Image recreation is necessary for text, wrap, colour or font
            # changes, but a simple state change keeps the existing bitmap.
            relevant={'text','font','foreground','fg','background','bg','wraplength','justify','textvariable'}
            if relevant.intersection(values):values,_=adapt(self,values)
            return _original(self,values)

        cls.__init__=constructor;cls.configure=configure;cls.config=configure

    def dialog(title=None,message=None,kind='info',**options):
        parent=options.get('parent') or tk._default_root
        own_root=None
        if parent is None:
            own_root=tk.Tk();own_root.withdraw();parent=own_root
        window=tk.Toplevel(parent);window.title(title or '확인')
        # Tk automatically withdraws a transient whose parent is withdrawn.
        # Native OpenCV tools use hidden Tk roots for dialogs, so only establish
        # a transient relationship when that parent is actually visible.
        if parent.winfo_viewable():window.transient(parent)
        window.resizable(False,False)
        body=tk.Frame(window,padx=22,pady=18);body.pack(fill='both',expand=True)
        text=str(message or '')
        if options.get('detail'):text+='\n\n'+str(options['detail'])
        tk.Label(body,text=text,font=('',14),wraplength=650,justify='left').pack(anchor='w',pady=(0,18))
        controls=tk.Frame(body);controls.pack(fill='x')
        result=[]
        def finish(value):result.append(value);window.destroy()
        if kind in ('yesno','okcancel'):
            positive='예' if kind=='yesno' else '확인'
            negative='아니오' if kind=='yesno' else '취소'
            tk.Button(controls,text=negative,font=('',13),command=lambda:finish(False),padx=18,pady=7).pack(side='right',padx=5)
            button=tk.Button(controls,text=positive,font=('',13),command=lambda:finish(True),padx=18,pady=7)
            button.pack(side='right',padx=5)
            window.protocol('WM_DELETE_WINDOW',lambda:finish(False))
        else:
            button=tk.Button(controls,text='확인',font=('',13),command=lambda:finish('ok'),padx=18,pady=7)
            button.pack(side='right');window.protocol('WM_DELETE_WINDOW',lambda:finish('ok'))
        window.bind('<Escape>',lambda _:finish(False if kind in ('yesno','okcancel') else 'ok'))
        window.bind('<Return>',lambda _:finish(True if kind in ('yesno','okcancel') else 'ok'))
        window.update_idletasks()
        width,height=window.winfo_reqwidth(),window.winfo_reqheight()
        x=max(0,(window.winfo_screenwidth()-width)//2);y=max(0,(window.winfo_screenheight()-height)//2)
        window.geometry(f'+{x}+{y}')
        previous_grab=parent.grab_current()
        try:
            window.grab_set()
            window._modal_grab_available=True
        except tk.TclError:
            # Another application can already own an X grab. Do not disturb
            # that application or crash this readable dialog; wait_window still
            # blocks this caller until its own explicit button response.
            window._modal_grab_available=False
        button.focus_set()
        parent.wait_window(window)
        if previous_grab is not None:
            try:previous_grab.grab_set()
            except tk.TclError:pass
        if own_root is not None:own_root.destroy()
        return result[0] if result else False

    for name,kind in [('askyesno','yesno'),('askokcancel','okcancel'),('showinfo','info'),
                      ('showerror','error'),('showwarning','warning')]:
        def call(title=None,message=None,_kind=kind,**options):return dialog(title,message,_kind,**options)
        setattr(messagebox,name,call)
    _INSTALLED=True
