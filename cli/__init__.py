"""命令行入口"""

from __future__ import annotations

import json
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()


def app() -> None:
    """CLI 主入口 — ship-hull 命令。"""
    args = sys.argv[1:]

    if args and args[0] in ("-h", "--help"):
        console.print(Panel(
            "[bold]ship-hull[/bold] — 船弦号识别 Agent\n\n"
            "用法:\n"
            "  ship-hull \"查询内容\"             单次查询\n"
            "  ship-hull --verbose \"查询内容\"    详细模式（显示工具调用链）\n"
            "  ship-hull --interactive / -i      交互模式\n"
            "  ship-hull --help / -h             帮助信息",
            title="帮助",
        ))
        return

    verbose = "--verbose" in args or "-v" in args
    interactive = "-i" in args or "--interactive" in args
    query_args = [a for a in args if a not in ("-i", "--interactive", "--verbose", "-v")]

    from agent import create_agent
    agent = create_agent()

    if interactive:
        _repl(agent, verbose)
    elif query_args:
        query = " ".join(query_args)
        _single_query(agent, query, verbose)
    else:
        console.print("[yellow]请提供查询内容，或使用 --interactive 进入交互模式。[/yellow]")
        console.print("用法: ship-hull \"查询内容\"  或  ship-hull -i")


def _single_query(agent, query: str, verbose: bool = False) -> None:
    """执行单次查询并打印结果。"""
    with console.status("[bold green]正在识别…"):
        if verbose:
            trace = agent.run_verbose(query)
            _print_trace(trace)
        else:
            answer = agent.run(query)
            console.print(Panel(answer, title="识别结果"))


def _print_trace(trace: list[dict]) -> None:
    """打印详细调用链。"""
    table = Table(title="🔧 Agent 调用链", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("类型", style="cyan", width=10)
    table.add_column("内容", style="white")

    for i, step in enumerate(trace):
        stype = step["type"]
        content = step.get("content", "")

        if "tool_calls" in step:
            calls = "; ".join(
                f"[green]{tc['name']}[/green]({json.dumps(tc['args'], ensure_ascii=False)})"
                for tc in step["tool_calls"]
            )
            content = f"→ {calls}"
        elif stype == "tool":
            content = f"← {content[:200]}"

        table.add_row(str(i), stype, content or "—")

    console.print(table)
    # 最后一条 AI 消息是最终回答
    for step in reversed(trace):
        if step["type"] == "ai" and step["content"]:
            console.print(Panel(step["content"], title="✅ 最终回答"))
            return


def _repl(agent, verbose: bool = False) -> None:
    """交互式 REPL。"""
    console.print(Panel(
        "[bold]船弦号识别 Agent[/bold] — 交互模式\n"
        "输入弦号或船只描述进行查询，输入 [bold red]quit[/bold red] 退出。\n"
        f"详细模式: {'[green]开启[/green]' if verbose else '[dim]关闭[/dim]'}（用 --verbose 开启）",
        title="🚢 Ship Hull Agent",
    ))

    while True:
        try:
            query = Prompt.ask("\n[bold cyan]查询[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]再见！[/yellow]")
            break

        if not query or query.strip().lower() in ("quit", "exit", "q"):
            console.print("[yellow]再见！[/yellow]")
            break

        _single_query(agent, query, verbose)
