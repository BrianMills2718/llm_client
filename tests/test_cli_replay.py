from __future__ import annotations

import argparse
import json

import llm_client
from llm_client.cli.replay import register_parser


def test_register_parser_sets_compare_handler() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parser(subparsers)
    args = parser.parse_args(
        ["replay", "compare", "--left-call-id", "1", "--right-call-id", "2"]
    )
    assert args.command == "replay"
    assert args.replay_command == "compare"
    assert callable(args.handler)


def test_register_parser_sets_rerun_handler() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parser(subparsers)
    args = parser.parse_args(
        ["replay", "rerun", "--call-id", "7", "--trace-id", "trace.replay"]
    )
    assert args.command == "replay"
    assert args.replay_command == "rerun"
    assert callable(args.handler)
    assert args.max_budget == 0.0


def test_cmd_replay_compare_text_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        llm_client,
        "compare_call_snapshots",
        lambda left, right: {"left_call_id": left, "right_call_id": right, "fingerprints_match": False, "request_differences": ["x"], "result_differences": []},
    )
    monkeypatch.setattr(llm_client, "format_call_diff", lambda report: "diff output")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parser(subparsers)
    args = parser.parse_args(
        ["replay", "compare", "--left-call-id", "1", "--right-call-id", "2"]
    )
    args.handler(args)

    assert capsys.readouterr().out.strip() == "diff output"


def test_cmd_replay_rerun_json_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        llm_client,
        "replay_call_snapshot",
        lambda call_id, **kwargs: {
            "source_call_id": call_id,
            "replay_trace_id": kwargs["trace_id"],
            "task": kwargs["task"],
            "project": kwargs["project"],
            "public_api": "call_llm",
            "result": "ignored",
        },
    )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parser(subparsers)
    args = parser.parse_args(
        [
            "replay",
            "rerun",
            "--call-id",
            "7",
            "--trace-id",
            "trace.replay",
            "--task",
            "task.replay",
            "--project",
            "project.replay",
            "--format",
            "json",
        ]
    )
    args.handler(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["source_call_id"] == 7
    assert payload["replay_trace_id"] == "trace.replay"
    assert payload["task"] == "task.replay"
