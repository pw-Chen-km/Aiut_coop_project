"""Real Docling PDF conversion, source-preserving export and normalization."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from .chunking import build_chunks
from .structure import build_sections, make_source_units, ordered_items, recover_native_pdf_cells, stable_id


class ParsingError(RuntimeError):
    pass


def parser_identity(parser_config: dict, versions: dict) -> str:
    """Chunking/model-retrieval parameters deliberately do not affect source IDs."""
    return stable_id("parse", "docling-canonical-v3-document-index", parser_config, versions)


def validate_page_count(raw: dict, document: dict) -> int | None:
    expected = document.get("expected_page_count", document.get("pages", document.get("page_count")))
    if expected is not None:
        if not isinstance(expected, int) or expected < 1:
            raise ParsingError("Expected manifest page count must be a positive integer")
        actual = len(raw.get("pages", {}))
        if actual != expected:
            raise ParsingError(f"Manifest page count mismatch for {document['doc_id']}: expected {expected}, parsed {actual}")
    return expected


def normalize_docling_json(raw: dict, document: dict, parse_version: str, *,
                           source_layout=False, fragment_repairs=(), reviewed_items=None, reviewed_audit=None) -> dict:
    expected_pages = validate_page_count(raw, document)
    items, warnings = ordered_items(raw)
    layout_audit = None
    if source_layout:
        from .comparison_source import repair_items
        items, layout_audit = repair_items(raw, fragment_repairs)
    if reviewed_items is not None:
        items, layout_audit = reviewed_items, reviewed_audit
    sections, blocks, figures, furniture = build_sections(items, document, parse_version)
    source_units = make_source_units(blocks)
    ref_blocks: dict[str, list[str]] = {}
    for block in blocks:
        ref_blocks.setdefault(block["native_ref"], []).append(block["block_id"])
    for figure in figures:
        figure["caption_block_ids"] = [bid for ref in figure["caption_refs"]
                                       for bid in ref_blocks.get(ref.get("$ref", ref.get("cref", "")), [])]
    pages = raw.get("pages", {})
    doc = {**document, "parse_version": parse_version, "page_count": len(pages), "expected_page_count": expected_pages,
           "docling_schema_version": raw.get("version")}
    return {"documents": [doc], "sections": sections, "blocks": blocks, "source_units": source_units,
            "chunks": [], "figures": figures, "furniture": furniture,
            "audit": {"doc_id": doc["doc_id"], "parse_version": parse_version, "warnings": warnings,
                      **({'source_layout': layout_audit} if layout_audit is not None else {}),
                      "native_items": len(items), "blocks": len(blocks), "source_units": len(source_units),
                      "sections": len(sections), "figures": len(figures), "furniture": len(furniture),
                      "zero_text_blocks": sum(not b["text"].strip() for b in blocks),
                      "missing_page_provenance": [b["block_id"] for b in blocks if b["page_index"] is None],
                      "source_offsets": "half-open character offsets in immutable canonical block.text",
                      "visual_evidence_status": "unreviewed; figures retained, not represented as understood text"}}


def _configure_pipeline(options: Any, parser_config: dict) -> Any:
    if not hasattr(options, "heading_hierarchy_options"):
        raise ParsingError("Installed Docling lacks heading_hierarchy_options; install a revision with native heading hierarchy support. No heuristic substitute is used.")
    options.heading_hierarchy_options.enabled = True
    options.generate_parsed_pages = True  # required for native style hierarchy
    if parser_config.get("heading_hierarchy", True) is not True:
        raise ParsingError("Native heading hierarchy must remain enabled")
    settings = {"do_ocr": False, "do_table_structure": True, "generate_picture_images": True,
                "generate_page_images": True}
    for name in ("do_ocr", "do_table_structure", "generate_picture_images", "generate_page_images",
                 "generate_parsed_pages", "artifacts_path", "document_timeout"):
        if name in parser_config:
            settings[name] = parser_config[name]
    settings.update(parser_config.get("pipeline_options", {}))
    if settings.get("generate_parsed_pages") is False:
        raise ParsingError("generate_parsed_pages=False disables heading style inference and is not allowed")
    for name, value in settings.items():
        if name == "heading_hierarchy_options":
            for subkey, subvalue in value.items():
                if subkey == "enabled" and subvalue is not True:
                    raise ParsingError("Native heading hierarchy must remain enabled")
                if not hasattr(options.heading_hierarchy_options, subkey):
                    raise ParsingError(f"Unsupported Docling heading option: {subkey}")
                setattr(options.heading_hierarchy_options, subkey, subvalue)
        else:
            if not hasattr(options, name):
                raise ParsingError(f"Unsupported Docling PDF pipeline option: {name}")
            setattr(options, name, value)
    return options


def parse_pdf(path: str | Path, document: dict, config: dict, out_dir: str | Path, tokenizer: Any) -> dict:
    path, out_dir = Path(path), Path(out_dir)
    if not path.is_file():
        raise ParsingError(f"PDF not found: {path}")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if document.get("source_sha256") and document["source_sha256"] != actual_hash:
        raise ParsingError(f"Source SHA-256 mismatch for {document['doc_id']}")
    document = {**document, "source_sha256": actual_hash, "source_path": str(path.resolve())}
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc import ContentLayer, ImageRefMode
    except ImportError as exc:
        raise ParsingError("Genuine Docling PDF conversion is required. Install the doc2skill PDF dependencies; no fallback parser is used.") from exc
    parser_config = dict(config.get("parser", {}))
    options = _configure_pipeline(PdfPipelineOptions(), parser_config)
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    device = parser_config.get("device", "cpu")
    try:
        options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice(device), num_threads=int(parser_config.get("threads", 4)))
    except (ValueError, TypeError) as exc:
        raise ParsingError(f"Unsupported parser accelerator configuration: {device}") from exc
    versions = {}
    for package in ("docling", "docling-core", "docling-parse", "docling-ibm-models"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    resolved_options = options.model_dump(mode="json")
    parse_version = parser_identity({"requested": parser_config, "resolved": resolved_options}, versions)
    converter = DocumentConverter(allowed_formats=[InputFormat.PDF], format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    converted = converter.convert(path)
    status = getattr(converted.status, "value", str(converted.status))
    if status != "success":
        raise ParsingError(f"Docling conversion did not fully succeed ({status}); refusing partial corpus")
    dl_doc = converted.document
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "docling.json", out_dir / "docling.md"
    # Complete JSON includes page/picture images, all tables/cells and native refs.
    dl_doc.save_as_json(json_path, image_mode=ImageRefMode.REFERENCED, artifacts_dir=out_dir / "images")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    validate_page_count(raw, document)
    dl_doc.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED, artifacts_dir=out_dir / "images",
                            included_content_layers=set(ContentLayer), traverse_pictures=True,
                            page_break_placeholder="<!-- page break -->")
    fragment = normalize_docling_json(raw, document, parse_version)
    native_pages = []
    for page in converted.pages:
        parsed_page = page.parsed_page
        native_pages.append({"page_no": page.page_no,
                             "height": page.size.height if page.size else None,
                             "width": page.size.width if page.size else None,
                             "parsed_page": parsed_page.model_dump(mode="json", exclude={"image"})
                                            if parsed_page is not None else None})
    # This separate complete native-cell registry is essential: assembled
    # Docling JSON does not contain backend cells discarded by table/layout models.
    native_path = out_dir / "native_pdf_pages.json"
    native_path.write_text(json.dumps({"schema_version": "native-pdf-pages-v1", "doc_id": document["doc_id"],
                                      "source_sha256": actual_hash, "pages": native_pages},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    recover_native_pdf_cells(fragment, native_pages)
    fragment["documents"][0].update({"docling_json_path": str(json_path.resolve()),
                                      "docling_markdown_path": str(md_path.resolve()), "parser_packages": versions,
                                      "parser_options": resolved_options,
                                      "raw_export_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest()})
    fragment["documents"][0].update({"native_pdf_pages_path": str(native_path.resolve()),
                                      "native_pdf_pages_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
                                      "extraction_method": "docling-native-hierarchy-with-audited-native-text-recovery"})
    fragment["chunks"] = build_chunks(dl_doc, fragment, config, tokenizer)
    fragment["audit"].update({"chunks": len(fragment["chunks"]), "conversion_status": status,
                               "native_heading_hierarchy": True, "parser_packages": versions,
                               "raw_export": str(json_path.resolve()), "markdown_export": str(md_path.resolve())})
    return fragment
