import csv
import uuid
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("File Service")
FILE = Path("demo.csv")


def _ensure():
    if not FILE.exists():
        FILE.write_text("id,name,qty\n", encoding="utf-8")


@mcp.tool()
def read_csv() -> list[dict]:
    _ensure()
    with FILE.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@mcp.tool()
def add_row(name: str, qty: int) -> dict:
    _ensure()
    row = {"id": uuid.uuid4().hex[:6], "name": name, "qty": str(qty)}
    with FILE.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=row.keys()).writerow(row)
    return row


@mcp.tool()
def update_qty(row_id: str, qty: int) -> str:
    _ensure()
    rows = read_csv()
    for r in rows:
        if r["id"] == row_id:
            r["qty"] = str(qty)
    with FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "qty"])
        w.writeheader()
        w.writerows(rows)
    return "ok"


@mcp.tool()
def delete_row(row_id: str) -> str:
    _ensure()
    rows = [r for r in read_csv() if r["id"] != row_id]
    with FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "qty"])
        w.writeheader()
        w.writerows(rows)
    return "ok"


if __name__ == "__main__":
    mcp.run()
