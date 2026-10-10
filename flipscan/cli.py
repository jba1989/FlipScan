"""FlipScan CLI."""

from __future__ import annotations

from pathlib import Path

import click

from .config import load_config
from .workspace import STAGES, Workspace


@click.group()
@click.version_option()
def main():
    """FlipScan: slow-mo book flip video -> EPUB."""


# ---------------------------------------------------------------- init

@main.command()
@click.argument("directory", type=click.Path(path_type=Path))
@click.option("--video", "videos", multiple=True, required=True,
              type=click.Path(exists=True, path_type=Path),
              help="來源影片。可重複指定以加入多部影片；出現在"
                   "多部影片中的頁面會自動合併（保留最佳拍攝畫面）。")
@click.option("--direction", "directions", multiple=True,
              type=click.Choice(["forward", "reverse"]),
              help="每個 --video 的可選翻頁方向提示（預設：forward；"
                   "會根據頁面比對結果與印刷頁碼自動修正順序）。")
@click.option("--reverse", is_flag=True, help="簡記：單一影片從後往前拍攝。")
@click.option("--title", default=None, help="書籍標題（EPUB 中繼資料）。")
@click.option("--expected-pages", type=int, default=None,
              help="用於缺頁偵測的預期頁數。")
def init(directory: Path, videos, directions, reverse, title, expected_pages):
    """Create a workspace: copy videos in, probe fps, write manifest.json."""
    if directions and len(directions) != len(videos):
        raise click.UsageError("--direction 必須為每個 --video 指定一次（或完全不指定）")

    from .project import create_project
    specs = [
        {
            "path": str(src),
            "direction": directions[i] if directions
                         else ("reverse" if reverse else "forward"),
        }
        for i, src in enumerate(videos)
    ]
    ws = create_project(directory, specs, title=title,
                        expected_pages=expected_pages, log=click.echo)
    click.echo(f"工作區已就緒：{ws.root} — 下一步：flipscan run {ws.root}")


# ---------------------------------------------------------------- addvideo

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option("--direction", type=click.Choice(["forward", "reverse"]), default="forward")
@click.option("--upside-down", is_flag=True, help="影片拍攝時旋轉了 180 度。")
def addvideo(directory: Path, video: Path, direction: str, upside_down: bool):
    """Add another capture video — shared pages merge, new pages slot in.

    Keep adding videos until every page is covered; duplicate captures collapse
    via printed page numbers after transcription."""
    ws = Workspace.open(directory)
    from .project import add_video
    add_video(ws, video, direction=direction, rotate=180 if upside_down else 0,
              log=click.echo)
    click.echo(f"影片已加入 — 執行 `flipscan run {directory}` 以合併進去")


# ---------------------------------------------------------------- run

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--stage", "only_stage", type=click.Choice(STAGES), default=None,
              help="執行單一階段（代表將重新執行該階段）。")
@click.option("--force", is_flag=True, help="即使已完成仍重新執行階段。")
@click.option("--provider", type=click.Choice(["ollama", "anthropic", "hybrid", "mock"]),
              default=None)
@click.option("--model", default=None, help="覆寫辨識模型名稱。")
@click.option("--ollama-url", default=None)
def run(directory: Path, only_stage, force, provider, model, ollama_url):
    """Run the pipeline (extract -> ... -> assemble), resuming where it left off."""
    ws = Workspace.open(directory)
    cfg = load_config(ws.root)
    if provider:
        cfg["provider"]["name"] = provider
    if ollama_url:
        cfg["provider"]["ollama_url"] = ollama_url
    if model:
        key = "anthropic_model" if cfg["provider"]["name"] == "anthropic" else "ollama_model"
        cfg["provider"][key] = model

    from .project import run_pipeline
    run_pipeline(ws, cfg, only_stage=only_stage, force=force, log=click.echo)


# ---------------------------------------------------------------- review

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
def review(directory: Path):
    """Generate the HTML review page (frame vs markdown + reshoot list)."""
    ws = Workspace.open(directory)
    from .review import generate_review, reshoot_list
    out = generate_review(ws, log=click.echo)
    items = reshoot_list(ws)
    if items:
        click.echo(f"重拍清單 ({len(items)} 頁)："
                   + ", ".join(i["id"] for i in items))
    else:
        click.echo("重拍清單：無 — 所有頁面看起來都正常")
    click.echo(f"open {out}")


# ---------------------------------------------------------------- patch

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--page", "page_id", required=True, help="頁面 ID，例如 p0142")
@click.argument("image", type=click.Path(exists=True, path_type=Path))
def patch(directory: Path, page_id: str, image: Path):
    """Replace a page's capture with a re-shot photo and re-process it."""
    import shutil

    ws = Workspace.open(directory)
    cfg = load_config(ws.root)
    page = ws.page(page_id)
    if page is None:
        raise click.UsageError(f"{ws.root} 中找不到頁面 {page_id!r}")

    patches = ws.root / "patches"
    patches.mkdir(exist_ok=True)
    dest = patches / f"{page_id}{image.suffix.lower()}"
    shutil.copy2(image, dest)
    page["patched_source"] = f"patches/{dest.name}"
    page["status"] = "patched"
    for key in ("md", "confidence", "flags", "transcribe_error"):
        page.pop(key, None)
    page["md"] = None

    click.echo(f"{page_id}: 正在前處理替換照片")
    from .stages.preprocess import preprocess_page
    preprocess_page(ws, page, cfg)
    ws.save()

    click.echo(f"{page_id}: transcribing")
    from .stages.transcribe import run as transcribe_run
    transcribe_run(ws, cfg, log=click.echo)

    ws.stage_reset("figures")  # re-run figures + assemble with the new page
    click.echo(f"{page_id}: 已完成補拍 — 請先執行 `flipscan run {directory}`，"
               f"再執行 `flipscan build {directory}` 以重新建置輸出檔")


# ---------------------------------------------------------------- addpage

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option("--position", default="end",
              help='"start"、"end" 或頁面索引（預設：end）')
@click.option("--cover", is_flag=True,
              help="將此照片作為書籍封面（EPUB 封面圖片，"
                   "不包含在內文中）")
def addpage(directory: Path, image: Path, position: str, cover: bool):
    """Add a page from a photo — covers, inside-cover text, or missed pages."""
    ws = Workspace.open(directory)
    cfg = load_config(ws.root)
    from .project import add_page_from_photo
    page = add_page_from_photo(ws, cfg, image, position=position,
                               role="cover" if cover else None, log=click.echo)
    click.echo(f"{page['id']} 已加入 — 請先執行 `flipscan run {directory}`，"
               f"再執行 `flipscan build {directory}` 以重新建置輸出檔")


# ---------------------------------------------------------------- delpage

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.argument("page_id")
@click.option("--restore", is_flag=True, help="還原先前已刪除的頁面。")
def delpage(directory: Path, page_id: str, restore: bool):
    """Delete an erroneous page from the book (soft: restorable, survives re-runs)."""
    ws = Workspace.open(directory)
    from .project import set_page_deleted
    try:
        set_page_deleted(ws, page_id, not restore)
    except KeyError:
        raise click.UsageError(f"找不到頁面 {page_id!r}")
    click.echo(f"{page_id} {'已還原' if restore else '已刪除'} — "
               f"請執行 `flipscan build {directory}` 以重新建置輸出檔")


# ---------------------------------------------------------------- build

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None,
              help="輸出檔案（預設：out/<workspace>.<ext>）")
@click.option("--format", "formats", multiple=True,
              type=click.Choice(["epub", "pdf", "pdf-latex", "pdf-facsimile"]),
              help="輸出格式；可重複指定（預設：epub）。pdf-latex 為"
                   "高品質 pandoc+XeLaTeX PDF（系統需先安裝這些工具）。")
@click.option("--title", default=None)
@click.option("--author", default=None)
@click.option("--device", default="none",
              type=click.Choice(["none", "xteink-x3", "xteink-x4", "eink-6in",
                                 "remarkable-2", "tablet"]),
              help="目標閱讀器的尺寸（圖片 + 重排/LaTeX PDF 頁面）。")
def build(directory: Path, output, formats, title, author, device):
    """Build the book (epub / pdf / pdf-latex / pdf-facsimile) from markdown."""
    ws = Workspace.open(directory)
    formats = formats or ("epub",)
    for fmt in formats:
        ext = "epub" if fmt == "epub" else "pdf"
        if output and len(formats) == 1:
            out = output
        else:
            suffix = {"pdf-facsimile": "-facsimile", "pdf-latex": "-latex"}.get(fmt, "")
            if device != "none":
                suffix += f"-{device}"
            out = ws.dir("out") / f"{ws.root.name}{suffix}.{ext}"
        if fmt == "epub":
            from .build_epub import build_epub
            build_epub(ws, out, title=title, author=author, device=device,
                       log=click.echo)
        elif fmt == "pdf-facsimile":
            from .build_pdf import build_pdf_facsimile
            build_pdf_facsimile(ws, out, title=title, device=device, log=click.echo)
        elif fmt == "pdf-latex":
            from .build_pdf_latex import build_pdf_latex
            build_pdf_latex(ws, out, title=title, author=author, device=device,
                            log=click.echo)
        else:
            from .build_pdf import build_pdf_reflowed
            build_pdf_reflowed(ws, out, title=title, device=device, log=click.echo)


# ---------------------------------------------------------------- status

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
def status(directory: Path):
    """Show pipeline stage status and page counts."""
    ws = Workspace.open(directory)
    for stage in STAGES:
        click.echo(f"{stage:12s} {ws.stage_status(stage)}")
    pages = ws.manifest["pages"]
    if pages:
        suspects = [p["id"] for p in pages if p.get("status") == "suspect"]
        click.echo(f"pages: {len(pages)} ({len(suspects)} suspect)")


# ---------------------------------------------------------------- ui

@main.command()
@click.option("--root", type=click.Path(path_type=Path), default=None,
              help="包含專案工作區的目錄（預設：FLIPSCAN_ROOT 或目前目錄）")
@click.option("--host", default="0.0.0.0",
              help="繫結位址（預設 0.0.0.0 = 區域網路內其他裝置可存取；"
                   "若僅限本機存取請使用 127.0.0.1）")
@click.option("--port", type=int, default=8321)
def ui(root, host, port):
    """Start the local web GUI (requires `pip install flipscan[ui]`)."""
    import os
    import socket

    try:
        from .ui import serve
    except ImportError:
        raise click.ClickException(
            "缺少 GUI 相依套件 — 請使用以下指令安裝：pip install 'flipscan[ui]'")
    root = root or Path(os.environ.get("FLIPSCAN_ROOT", "."))
    root.mkdir(parents=True, exist_ok=True)   # ensure the projects folder exists
    from .ui.security import load_or_create_token
    tok = f"?token={load_or_create_token(root)}"
    click.echo(f"FlipScan GUI  (專案根目錄：{root.resolve()})")
    # token included: inside Docker even the host's own browser isn't loopback
    click.echo(f"  本機：        http://localhost:{port}/{tok}")
    if host == "0.0.0.0":
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            lan_ip = s.getsockname()[0]
            s.close()
            click.echo(f"  區域網路：    http://{lan_ip}:{port}/{tok}  (手機、平板等…)")
            click.echo(f"  手機麥克風：  https://{lan_ip}:{port + 1}/{tok}  "
                       f"(語音錄音需要 https — 請接受一次"
                       f"安全性憑證警告)")
            click.echo("  權杖連結即為其他裝置的存取密碼 — "
                       "請勿分享給不信任的人員")
        except OSError:
            pass
    serve(root, host=host, port=port)


# ---------------------------------------------------------------- worker

@main.command()
@click.option("--root", type=click.Path(path_type=Path), default=None,
              help="包含專案工作區的目錄（預設：FLIPSCAN_ROOT 或目前目錄）")
def worker(root):
    """Run the durable background job worker as its own process.

    The pipeline, proofreads, and page re-reads run as durable jobs in a
    SQLite queue (`jobs.db` in the projects root). `flipscan ui` already runs
    an in-process worker, so you only need this to run the worker separately —
    e.g. a second docker-compose service — so restarting the web server never
    interrupts a running job. Point it at the SAME root as the web server and
    set FLIPSCAN_EXTERNAL_WORKER=1 on the web server so it doesn't double up.
    """
    import os
    import time

    from .jobs import JobQueue
    from .jobs_handlers import concurrency_config, register_handlers

    root = root or Path(os.environ.get("FLIPSCAN_ROOT", "."))
    root.mkdir(parents=True, exist_ok=True)
    lane_caps, kind_lanes = concurrency_config()
    jobq = JobQueue(root / "jobs.db", lane_caps=lane_caps, kind_lanes=kind_lanes)
    register_handlers(jobq, root)
    resumed = jobq.requeue_orphans()
    click.echo(f"FlipScan 背景工作處理器  (專案根目錄：{root.resolve()})")
    click.echo(f"  已繼續執行 {resumed} 個孤立工作；等待工作中…")
    jobq.start_worker()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        jobq.stop()
        click.echo("工作處理器已停止")


if __name__ == "__main__":
    main()
