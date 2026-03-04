"""Drag-and-drop web UI for AutoForge.

Run with:
    autoforge-ui

Or:
    python -m autoforge.gradio_ui
"""

from __future__ import annotations

import csv
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_MATERIALS_CSV = Path(__file__).resolve().parents[2] / "materials.csv"
DEFAULT_OUTPUT_DIR = Path.home() / "autoforge"


def _ensure_gradio():
    try:
        import gradio as gr  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Gradio is required for the drag-and-drop UI. "
            "Install with: pip install 'AutoForge[ui]' or pip install gradio"
        ) from exc


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _row_label(row: dict[str, str]) -> str:
    name = row.get("Name", "").strip()
    color = row.get("Color", "").strip()
    brand = row.get("Brand", "").strip()
    mat_type = row.get("Type", "").strip()
    return f"{name} ({color}) - {brand} {mat_type}".strip()


def _material_family(material_type: str) -> str:
    t = (material_type or "").strip().upper()
    if t.startswith("PLA"):
        return "PLA Family"
    if t == "PETG":
        return "PETG"
    return t or "Unknown"


def _load_material_rows() -> tuple[list[dict[str, str]], list[str]]:
    if not DEFAULT_MATERIALS_CSV.exists():
        return [], []
    rows: list[dict[str, str]] = []
    with DEFAULT_MATERIALS_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_header = next(reader, None)
        if not raw_header:
            return [], []
        header = [h.strip() for h in raw_header]
        for record in reader:
            row: dict[str, str] = {}
            for i, key in enumerate(header):
                row[key] = record[i].strip() if i < len(record) else ""
            rows.append(row)
    labels = [_row_label(r) for r in rows]
    return rows, labels


def _group_labels_by_type(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        family = _material_family(row.get("Type", ""))
        grouped.setdefault(family, []).append(_row_label(row))
    for family in grouped:
        grouped[family] = sorted(grouped[family])
    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def _validate_material_mix(selected_labels: list[str]) -> str:
    rows, _ = _load_material_rows()
    type_by_label = {_row_label(r): r.get("Type", "") for r in rows}
    families = {
        _material_family(type_by_label[label])
        for label in selected_labels
        if label in type_by_label
    }
    if "PETG" in families and "PLA Family" in families:
        return (
            "Unsupported material mix: PETG and PLA-family filaments cannot be combined in one run. "
            "Select only PETG or only PLA-family materials."
        )
    return ""


def _build_filtered_materials_csv(selected_labels: list[str]) -> tuple[Optional[Path], str]:
    rows, _ = _load_material_rows()
    if not rows:
        return None, f"Missing or empty materials file: {DEFAULT_MATERIALS_CSV}"
    if not selected_labels:
        return None, "Please select at least one material color."

    selected_set = set(selected_labels)
    filtered_rows = [r for r in rows if _row_label(r) in selected_set]
    if not filtered_rows:
        return None, "None of the selected materials were found in materials.csv."

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="autoforge_selected_", delete=False, encoding="utf-8", newline=""
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=list(filtered_rows[0].keys()))
        writer.writeheader()
        writer.writerows(filtered_rows)
        return Path(tmp.name), ""


def run_pipeline(
    input_image: str,
    selected_materials: Optional[list[str]],
    priority_mask: Optional[str],
    iterations: int,
    max_layers: int,
    pruning_max_swaps: int,
    stl_output_size: int,
    flatforge: bool,
) -> Tuple[Optional[str], Optional[str], str]:
    if not input_image:
        return None, None, "Please upload an input image."
    if not DEFAULT_MATERIALS_CSV.exists():
        return (
            None,
            None,
            f"Missing materials file at: {DEFAULT_MATERIALS_CSV}. "
            "Add materials.csv to the repository root.",
        )

    filtered_csv_path, csv_error = _build_filtered_materials_csv(selected_materials or [])
    if csv_error:
        return None, None, csv_error
    mix_error = _validate_material_mix(selected_materials or [])
    if mix_error:
        return None, None, mix_error
    interpreted_max_colors = max(2, len(selected_materials or []))

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "autoforge.auto_forge",
        "--input_image",
        input_image,
        "--output_folder",
        str(output_dir),
        "--iterations",
        str(iterations),
        "--max_layers",
        str(max_layers),
        "--pruning_max_colors",
        str(interpreted_max_colors),
        "--pruning_max_swaps",
        str(pruning_max_swaps),
        "--stl_output_size",
        str(stl_output_size),
        "--no-visualize",
        "--disable_visualization_for_gradio",
        "1",
    ]

    cmd.extend(["--csv_file", str(filtered_csv_path)])

    if priority_mask:
        cmd.extend(["--priority_mask", priority_mask])
    if flatforge:
        cmd.append("--flatforge")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
    finally:
        if filtered_csv_path is not None and filtered_csv_path.exists():
            filtered_csv_path.unlink(missing_ok=True)

    logs = "\n".join(
        [
            f"Command: {' '.join(shlex.quote(x) for x in cmd)}",
            "",
            "STDOUT:",
            proc.stdout.strip() or "<empty>",
            "",
            "STDERR:",
            proc.stderr.strip() or "<empty>",
        ]
    )

    if proc.returncode != 0:
        return None, None, f"AutoForge failed (exit code {proc.returncode}).\n\n{logs}"

    preview = _first_existing(
        [
            output_dir / "final_model.png",
            output_dir / "priority_mask_resized.png",
        ]
    )
    downloadable = _first_existing(
        [
            output_dir / "final_model.stl",
            output_dir / "project_file.hfp",
            output_dir / "swap_instructions.txt",
            output_dir / "final_model.png",
        ]
    )
    return (
        str(preview) if preview else None,
        str(downloadable) if downloadable else None,
        (
            "AutoForge completed successfully.\n\n"
            f"Interpreted max colors from selection: {interpreted_max_colors}\n"
            f"Outputs saved to: {output_dir}\n\n{logs}"
        ),
    )


def build_app():
    _ensure_gradio()
    import gradio as gr
    material_rows, _ = _load_material_rows()
    grouped_labels = _group_labels_by_type(material_rows)

    def _run_from_tabs(
        input_image: str,
        priority_mask: Optional[str],
        iterations: int,
        max_layers: int,
        pruning_max_swaps: int,
        stl_output_size: int,
        flatforge: bool,
        *type_selections,
    ) -> Tuple[Optional[str], Optional[str], str]:
        merged: list[str] = []
        for selection in type_selections:
            if not selection:
                continue
            if isinstance(selection, str):
                merged.append(selection)
            else:
                merged.extend(selection)
        selected_materials = list(dict.fromkeys(merged))
        return run_pipeline(
            input_image=input_image,
            selected_materials=selected_materials,
            priority_mask=priority_mask,
            iterations=iterations,
            max_layers=max_layers,
            pruning_max_swaps=pruning_max_swaps,
            stl_output_size=stl_output_size,
            flatforge=flatforge,
        )

    with gr.Blocks(title="AutoForge Drag-and-Drop UI") as app:
        gr.Markdown("## AutoForge Drag-and-Drop Interface")
        gr.Markdown(
            f"Drop an image, choose materials by filament type tabs, then run AutoForge. Materials: `{DEFAULT_MATERIALS_CSV}`. Output folder: `{DEFAULT_OUTPUT_DIR}`."
        )

        with gr.Row():
            input_image = gr.Image(
                type="filepath",
                label="Input Image (drop here)",
            )
            priority_mask = gr.Image(
                type="filepath",
                label="Optional Priority Mask (drop here)",
            )

        material_selectors: list[gr.Dropdown] = []
        with gr.Tabs():
            for mat_type, labels in grouped_labels.items():
                with gr.Tab(f"{mat_type} ({len(labels)})"):
                    selector = gr.Dropdown(
                        choices=labels,
                        value=labels,
                        multiselect=True,
                        allow_custom_value=False,
                        filterable=True,
                        label=f"{mat_type} Materials",
                        info="Search and select colors to allow for this run.",
                    )
                    with gr.Row():
                        select_all_btn = gr.Button(f"Select All {mat_type}", size="sm")
                        clear_btn = gr.Button(f"Clear {mat_type}", size="sm")
                    select_all_btn.click(
                        fn=lambda all_labels=labels: all_labels,
                        outputs=[selector],
                    )
                    clear_btn.click(fn=lambda: [], outputs=[selector])
                    material_selectors.append(selector)

        with gr.Row():
            iterations = gr.Slider(
                minimum=100,
                maximum=10000,
                step=100,
                value=2000,
                label="Iterations",
            )
            max_layers = gr.Slider(
                minimum=10,
                maximum=200,
                step=1,
                value=75,
                label="Max Layers",
            )
            stl_output_size = gr.Slider(
                minimum=50,
                maximum=300,
                step=5,
                value=150,
                label="STL Size (mm)",
            )

        with gr.Row():
            pruning_max_swaps = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                value=20,
                label="Max Swaps",
            )
            flatforge = gr.Checkbox(value=False, label="FlatForge Mode")

        run_btn = gr.Button("Run AutoForge", variant="primary")
        gr.Markdown(
            "Compatibility rule: you can mix PLA variants together (PLA, PLA+, PLA Silk, etc.), but do not mix PETG with PLA-family materials."
        )

        preview_out = gr.Image(label="Output Preview")
        download_out = gr.File(label="Download Main Output")
        log_out = gr.Textbox(label="Run Log", lines=16)

        run_btn.click(
            fn=_run_from_tabs,
            inputs=[
                input_image,
                priority_mask,
                iterations,
                max_layers,
                pruning_max_swaps,
                stl_output_size,
                flatforge,
                *material_selectors,
            ],
            outputs=[preview_out, download_out, log_out],
        )

    return app


def main() -> None:
    app = build_app()
    app.launch(allowed_paths=[str(DEFAULT_OUTPUT_DIR)])


if __name__ == "__main__":
    main()
