from typing import List

from bokeh import embed, colors
from bokeh.plotting import figure, show

from common.const import COLORMAP

palette = [colors.RGB(*[int(c * 255) for c in color]) for color in COLORMAP]
darker_palette = [c.darken(0.3) for c in palette]

tooltip_css = """
tooltip {
    background-color: #212121;
    color: white;
    padding: 5px;
    border-radius: 10px;
    margin-left: 10px;
}
"""

plot_styling = dict(
    plot_height=500,
    sizing_mode="stretch_width",
    toolbar_location=None,
)


def style_plot(p):
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.outline_line_color = None

    p.axis.axis_label_text_font = "times"
    p.axis.axis_label_text_font_size = "12pt"
    p.axis.axis_label_text_font_style = "bold"

    p.legend.label_text_font = "times"
    p.legend.label_text_font_size = "12pt"
    p.legend.label_text_font_style = "normal"

    p.title.align = "center"
    p.title.text_font_size = "14pt"
    p.title.text_font = "Fira Sans"

    # p.y_range.start = 0


def plot_memory_usage(memory_trace: List):
    print((min(memory_trace), max(memory_trace)))
    x = list(range(len(memory_trace)))
    y = memory_trace

    p = figure(title="Memory Usage",
               x_axis_label='Cycle',
               y_axis_label='Memory in GB',
               y_range=(min(memory_trace), max(memory_trace)),
               **plot_styling)

    p.line(x, y, legend_label="RAM", line_width=2)

    p.legend.location = "bottom_right"
    style_plot(p)

    return embed.components(p)