"""Figures 4, 5 and 6 again, with use case 4 added.

The figures cover use cases 1-3 by default. This renders a second version of
each main figure over all four use cases, written alongside the defaults with
a `_w_u4` suffix.

Each figure's own `main()` takes the use cases to draw and a filename suffix,
so both versions come out of one code path: run a figure module directly for
the default set, run this for the extended one. The two sets are registered
in `make_fig4_config_insights.VARIANTS`.

Usage (ritme_usecases env), from the repo root or from this directory:
    python -m final_figures.make_w_u4
    python make_w_u4.py
"""

from __future__ import annotations

try:
    from final_figures import make_fig4_config_insights as fig4
    from final_figures import make_fig5_relation as fig5
    from final_figures import make_fig6_time_course as fig6
except ImportError:  # run as a plain script from inside final_figures/
    import make_fig4_config_insights as fig4
    import make_fig5_relation as fig5
    import make_fig6_time_course as fig6

SUFFIX = "_w_u4"
FIGURES = (fig4, fig5, fig6)


def main() -> None:
    use_cases = list(fig4.VARIANTS[SUFFIX])
    print(f"Drawing {', '.join(use_cases)} as {SUFFIX}")
    for module in FIGURES:
        module.main(use_cases=use_cases, suffix=SUFFIX)


if __name__ == "__main__":
    main()
