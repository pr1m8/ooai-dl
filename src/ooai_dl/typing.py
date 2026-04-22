"""Protocols for LangChain-style document producers.

Purpose
-------
Define structural typing contracts for objects that produce
``langchain_core.documents.Document`` instances.

Design
------
Instead of tying utility code to ``BaseLoader`` inheritance, this module
models loader *capabilities* with small protocols such as sync lazy load,
eager load, async lazy load, and async eager load.

These protocols are useful for:
- wrappers and adapters,
- utilities that should work with many concrete loaders,
- testing with fakes or stubs,
- structural typing when an object behaves like a loader but does not
  directly inherit from ``BaseLoader``.

Examples
--------
>>> from collections.abc import Iterator
>>> from langchain_core.documents import Document
>>>
>>> class FakeLoader:
...     def lazy_load(self) -> Iterator[Document]:
...         yield Document(page_content="hello")
...
>>> def consume(loader: SupportsLazyLoad[Document]) -> list[Document]:
...     return list(loader.lazy_load())
...
>>> docs = consume(FakeLoader())
>>> docs[0].page_content
'hello'
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

type DocumentSeq[DocT: Document] = Sequence[DocT]
type DocumentList[DocT: Document] = list[DocT]
type UrlSequence = Sequence[str]
type ParserName = str | None


@runtime_checkable
class SupportsLazyLoad[DocT: Document](Protocol):
    """Protocol for sync lazy document loaders.

    Examples
    --------
    >>> def read_all(loader: SupportsLazyLoad[Document]) -> list[Document]:
    ...     return list(loader.lazy_load())
    """

    def lazy_load(self) -> Iterator[DocT]:
        """Yield documents lazily.

        Returns:
            An iterator of documents.
        """


@runtime_checkable
class SupportsLoad[DocT: Document](Protocol):
    """Protocol for eager sync document loaders."""

    def load(self) -> DocumentList[DocT]:
        """Load all documents eagerly.

        Returns:
            A list of documents.
        """


@runtime_checkable
class SupportsAsyncLazyLoad[DocT: Document](Protocol):
    """Protocol for async lazy document loaders."""

    async def alazy_load(self) -> AsyncIterator[DocT]:
        """Yield documents lazily in async code.

        Returns:
            An async iterator of documents.
        """


@runtime_checkable
class SupportsAsyncLoad[DocT: Document](Protocol):
    """Protocol for eager async document loaders."""

    async def aload(self) -> DocumentList[DocT]:
        """Load all documents eagerly in async code.

        Returns:
            A list of documents.
        """


@runtime_checkable
class SupportsLoadAndSplit[DocT: Document](Protocol):
    """Protocol for loaders that support split-after-load behavior."""

    def load_and_split(self, text_splitter: Any | None = None) -> DocumentList[DocT]:
        """Load and optionally split documents.

        Args:
            text_splitter:
                Optional splitter object.

        Returns:
            A list of documents.
        """


@runtime_checkable
class DocumentLoaderLike[DocT: Document](
    SupportsLazyLoad[DocT],
    SupportsLoad[DocT],
    Protocol,
):
    """Composite protocol for basic sync LangChain-style loaders."""


@runtime_checkable
class AsyncDocumentLoaderLike[DocT: Document](
    SupportsAsyncLazyLoad[DocT],
    SupportsAsyncLoad[DocT],
    Protocol,
):
    """Composite protocol for basic async LangChain-style loaders."""


@runtime_checkable
class FullDocumentLoaderLike[DocT: Document](
    DocumentLoaderLike[DocT],
    AsyncDocumentLoaderLike[DocT],
    Protocol,
):
    """Composite protocol for loaders supporting both sync and async APIs."""


@runtime_checkable
class SupportsFetchAll[FetchResultT](Protocol):
    """Protocol for loaders that can batch-fetch URL content."""

    def fetch_all(self, urls: UrlSequence) -> FetchResultT:
        """Fetch raw content for many URLs.

        Args:
            urls:
                URLs to fetch.

        Returns:
            Raw fetched content in loader-specific form.
        """


@runtime_checkable
class SupportsScrape[ScrapeResultT](Protocol):
    """Protocol for loaders that can scrape a single resource."""

    def scrape(self, parser: ParserName = None) -> ScrapeResultT:
        """Scrape a single resource.

        Args:
            parser:
                Optional parser hint.

        Returns:
            Scraped result in loader-specific form.
        """


@runtime_checkable
class SupportsScrapeAll[ScrapeResultT](Protocol):
    """Protocol for loaders that can scrape multiple URLs."""

    def scrape_all(
        self,
        urls: UrlSequence,
        parser: ParserName = None,
    ) -> list[ScrapeResultT]:
        """Scrape multiple resources.

        Args:
            urls:
                URLs to scrape.
            parser:
                Optional parser hint.

        Returns:
            A list of scraped results.
        """


@runtime_checkable
class SupportsAsyncScrapeAll[ScrapeResultT](Protocol):
    """Protocol for loaders that can asynchronously scrape multiple URLs."""

    async def ascrape_all(
        self,
        urls: UrlSequence,
        parser: ParserName = None,
    ) -> list[ScrapeResultT]:
        """Scrape multiple resources asynchronously.

        Args:
            urls:
                URLs to scrape.
            parser:
                Optional parser hint.

        Returns:
            A list of scraped results.
        """


def consume_sync_loader[DocT: Document](
    loader: SupportsLazyLoad[DocT] | SupportsLoad[DocT],
) -> list[DocT]:
    """Normalize a sync loader into a document list.

    Args:
        loader:
            Any object supporting either ``lazy_load()`` or ``load()``.

    Returns:
        A list of documents.

    Raises:
        TypeError:
            If the object supports neither protocol.
    """
    if isinstance(loader, SupportsLazyLoad):
        return list(loader.lazy_load())
    if isinstance(loader, SupportsLoad):
        return loader.load()
    raise TypeError("Object does not support lazy_load() or load().")


async def consume_async_loader[DocT: Document](
    loader: SupportsAsyncLazyLoad[DocT] | SupportsAsyncLoad[DocT],
) -> list[DocT]:
    """Normalize an async loader into a document list.

    Args:
        loader:
            Any object supporting either ``alazy_load()`` or ``aload()``.

    Returns:
        A list of documents.

    Raises:
        TypeError:
            If the object supports neither protocol.
    """
    if isinstance(loader, SupportsAsyncLazyLoad):
        return [doc async for doc in loader.alazy_load()]
    if isinstance(loader, SupportsAsyncLoad):
        return await loader.aload()
    raise TypeError("Object does not support alazy_load() or aload().")


def accepts_true_langchain_loader(loader: BaseLoader) -> list[Document]:
    """Example of using the concrete LangChain base class directly.

    Args:
        loader:
            A real ``BaseLoader`` instance.

    Returns:
        A list of documents.
    """
    return loader.load()