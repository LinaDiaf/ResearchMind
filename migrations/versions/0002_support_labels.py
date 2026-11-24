"""Support labels for claims that were not checked or carry no citation.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-24
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE support_label ADD VALUE IF NOT EXISTS 'not_checked'")
    op.execute("ALTER TYPE support_label ADD VALUE IF NOT EXISTS 'uncited'")


def downgrade() -> None:
    # PostgreSQL cannot drop enum values; rows using them are mapped back first.
    op.execute("UPDATE report_claims SET support = 'partially_supported' "
               "WHERE support::text IN ('not_checked', 'uncited')")
    op.execute("ALTER TYPE support_label RENAME TO support_label_old")
    op.execute("CREATE TYPE support_label AS ENUM ('supported', 'partially_supported', 'unsupported')")
    op.execute("ALTER TABLE report_claims ALTER COLUMN support TYPE support_label "
               "USING support::text::support_label")
    op.execute("DROP TYPE support_label_old")
