"""create prediction persistence tables

Revision ID: 0001_create_prediction_records
Revises:
Create Date: 2026-05-10 18:20:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_create_prediction_records"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Версионирование данных
    op.create_table(
        "prediction_records",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("label", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("probabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("processing_time_ms", sa.Float(), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_prediction_records")),
    )
    op.create_index(
        "ix_prediction_records_created_at_desc",
        "prediction_records",
        [sa.text("created_at DESC")],
        unique=False,
    )
    op.create_index(
        "ix_prediction_records_label",
        "prediction_records",
        ["label"],
        unique=False,
    )

    op.create_table(
        "model_training_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("accuracy", sa.Float(), nullable=False),
        sa.Column("macro_f1", sa.Float(), nullable=False),
        sa.Column(
            "classification_report",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("train_size", sa.Integer(), nullable=False),
        sa.Column("test_size", sa.Integer(), nullable=False),
        sa.Column(
            "trained_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_training_runs")),
    )
    op.create_index(
        "ix_model_training_runs_trained_at_desc",
        "model_training_runs",
        [sa.text("trained_at DESC")],
        unique=False,
    )
    op.create_index(
        "ix_model_training_runs_model_identity",
        "model_training_runs",
        ["model_name", "model_version"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_model_training_runs_model_identity", table_name="model_training_runs")
    op.drop_index("ix_model_training_runs_trained_at_desc", table_name="model_training_runs")
    op.drop_table("model_training_runs")

    op.drop_index("ix_prediction_records_label", table_name="prediction_records")
    op.drop_index("ix_prediction_records_created_at_desc", table_name="prediction_records")
    op.drop_table("prediction_records")
