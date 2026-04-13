"""命令行入口"""

from __future__ import annotations

import sys
import logging

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def app() -> None:
    """CLI 主入口 — ship-hull 命令。"""
    args = sys.argv[1:]

    if args and args[0] in ("-h", "--help"):
        console.print(Panel(
            "[bold]ship-hull[/bold] — 船弦号识别 Agent\n\n"
            "用法:\n"
            "  ship-hull \"查询内容\"          单次查询\n"
            "  ship-hull --interactive / -i   交互模式\n"
            "  ship-hull --help / -h          帮助信息",
            title="帮助",
        ))
        return

    interactive = args and args[0] in ("-i", "--interactive")

    # 延迟导入，让 --help 更快响应
    from agent import create_agent

    agent = create_agent()

    if interactive:
        _repl(agent)
    elif args:
        query = " ".join(args)
        _single_query(agent, query)
    else:
        console.print("[yellow]请提供查询内容，或使用 --interactive 进入交互模式。[/yellow]")
        console.print("用法: ship-hull \"查询内容\"  或  ship-hull -i")


def _single_query(agent, query: str) -> None:
    """执行单次查询并打印结果。"""
    with console.status("[bold green]正在识别…"):
        answer = agent.run(query)
    console.print(Panel(answer, title="识别结果"))


def _repl(agent) -> None:
    """交互式 REPL。"""
    console.print(Panel(
        "[bold]船弦号识别 Agent[/bold] — 交互模式\n"
        "输入弦号或船只描述进行查询，输入 [bold red]quit[/bold red] 退出。",
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

        _single_query(agent, query)
