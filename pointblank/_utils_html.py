from __future__ import annotations

from pointblank._constants import TABLE_TYPE_STYLES


def _create_table_type_html(
    tbl_type: str | None, tbl_name: str | None, font_size: str = "smaller"
) -> str:

    if tbl_type is None:
        return ""

    style = TABLE_TYPE_STYLES.get(tbl_type)

    if style is None:
        return ""

    if tbl_name is None:
        return (
            f"<span style='background-color: {style['background']}; color: {style['text']}; padding: 0.5em 0.5em; "
            f"position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px {style['background']}; "
            f"font-weight: bold; padding: 2px 10px 2px 10px; font-size: {font_size};'>{style['label']}</span>"
        )

    return (
        f"<span style='background-color: {style['background']}; color: {style['text']}; padding: 0.5em 0.5em; "
        f"position: inherit; text-transform: uppercase; margin: 5px 0px 5px 0px; border: solid 1px {style['background']}; "
        f"font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;'>{style['label']}</span>"
        f"<span style='background-color: none; color: #222222; padding: 0.5em 0.5em; "
        f"position: inherit; margin: 5px 10px 5px -4px; border: solid 1px {style['background']}; "
        f"font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;'>{tbl_name}</span>"
    )


def _create_table_dims_html(columns: int, rows: int, font_size: str = "10px") -> str:

    return (
        f"<span style='background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; "
        f"position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; "
        f"font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; "
        f"font-size: {font_size};'>Rows</span>"
        f"<span style='background-color: none; color: #333333; padding: 0.5em 0.5em; "
        f"position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; "
        f"border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: {font_size};'>{rows}</span>"
        f"<span style='background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; "
        f"position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; "
        f"font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; "
        f"font-size: {font_size};'>Columns</span>"
        f"<span style='background-color: none; color: #333333; padding: 0.5em 0.5em; "
        f"position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; "
        f"border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: {font_size};'>{columns}</span>"
    )
