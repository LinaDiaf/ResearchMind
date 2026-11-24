"""Application wiring: builds the shared components once per process."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from researchmind.config import AgentBudgetSettings, Settings, get_budget_settings, get_settings
from researchmind.db.session import get_sessionmaker
from researchmind.log import configure_logging


@dataclass
class Services:
    settings: Settings
    budget: AgentBudgetSettings

    @cached_property
    def sessionmaker(self):
        return get_sessionmaker(self.settings.database_url)

    @cached_property
    def embedder(self):
        from researchmind.embeddings import build_embedder

        return build_embedder(self.settings)

    @cached_property
    def reranker(self):
        from researchmind.retrieval.reranker import build_reranker

        return build_reranker(self.settings)

    @cached_property
    def engine(self):
        from researchmind.retrieval.engine import RetrievalEngine

        return RetrievalEngine(self.sessionmaker, self.embedder, self.reranker, self.settings.index_dir)

    @cached_property
    def http(self):
        from researchmind.discovery.http import RateLimitedClient

        return RateLimitedClient(cache_dir=self.settings.cache_dir,
                                 contact_email=self.settings.contact_email)

    @cached_property
    def library(self):
        from researchmind.ingestion.service import Library

        return Library(self.sessionmaker, self.embedder, self.settings, self.http)

    @cached_property
    def discovery(self):
        from researchmind.discovery.sources import build_discovery

        return build_discovery(self.settings)

    @cached_property
    def provider(self):
        from researchmind.llm import build_provider

        return build_provider(self.settings)

    @cached_property
    def checkpointer(self):
        from researchmind.workspace.investigations import make_postgres_checkpointer

        return make_postgres_checkpointer(self.settings.psycopg_url)

    @cached_property
    def runner(self):
        from researchmind.workspace.investigations import InvestigationRunner

        return InvestigationRunner(self.sessionmaker, self.engine, self.provider, self.budget,
                                   self.checkpointer)

    @cached_property
    def workspace(self):
        from researchmind.workspace.investigations import Workspace

        return Workspace(self.sessionmaker)

    def comparator(self):
        from researchmind.comparison import Comparator
        from researchmind.llm import LLMClient

        return Comparator(self.engine.retrieve, LLMClient(self.provider))


def build_services(settings: Settings | None = None) -> Services:
    s = settings or get_settings()
    configure_logging(s.log_level, s.log_json)
    s.ensure_dirs()
    return Services(s, get_budget_settings())
