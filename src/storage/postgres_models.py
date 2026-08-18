"""SQLAlchemy ORM models for FRC competition data in PostgreSQL."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    """Declarative base for all PostgreSQL models."""


# Many-to-many association: teams participating in an event
event_teams = Table(
    "event_teams",
    Base.metadata,
    Column("event_id", Integer, ForeignKey("events.id", ondelete="CASCADE"), primary_key=True),
    Column("team_id", Integer, ForeignKey("teams.id", ondelete="CASCADE"), primary_key=True),
)


class Team(Base):
    """FRC team record."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_number: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    nickname: Mapped[str | None] = mapped_column(String(255))
    city: Mapped[str | None] = mapped_column(String(128))
    state_prov: Mapped[str | None] = mapped_column(String(64))
    country: Mapped[str | None] = mapped_column(String(64))
    website: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    events: Mapped[list[Event]] = relationship(
        secondary=event_teams, back_populates="teams"
    )
    red_matches: Mapped[list[Match]] = relationship(
        "Match",
        secondary="match_red_teams",
        back_populates="red_alliance",
    )
    blue_matches: Mapped[list[Match]] = relationship(
        "Match",
        secondary="match_blue_teams",
        back_populates="blue_alliance",
    )

    def __repr__(self) -> str:
        return f"<Team {self.team_number} '{self.nickname}'>"

    @classmethod
    def from_tba(cls, tba_team: dict) -> Team:
        """Build a Team instance from a TBA /team/{key} response."""
        return cls(
            team_number=tba_team["team_number"],
            nickname=tba_team.get("nickname"),
            city=tba_team.get("city"),
            state_prov=tba_team.get("state_prov"),
            country=tba_team.get("country"),
            website=tba_team.get("website"),
        )


class Event(Base):
    """FRC competition event."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_key: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    event_type: Mapped[str | None] = mapped_column(String(64))
    city: Mapped[str | None] = mapped_column(String(128))
    state_prov: Mapped[str | None] = mapped_column(String(64))
    country: Mapped[str | None] = mapped_column(String(64))
    start_date: Mapped[datetime | None] = mapped_column(DateTime)
    end_date: Mapped[datetime | None] = mapped_column(DateTime)
    website: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    teams: Mapped[list[Team]] = relationship(
        secondary=event_teams, back_populates="events"
    )
    matches: Mapped[list[Match]] = relationship(
        "Match", back_populates="event", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Event {self.event_key} '{self.name}'>"

    @classmethod
    def from_tba(cls, tba_event: dict) -> Event:
        """Build an Event instance from a TBA /event/{key} response."""
        return cls(
            event_key=tba_event["key"],
            name=tba_event.get("name"),
            year=tba_event.get("year"),
            event_type=str(tba_event.get("event_type", "")),
            city=tba_event.get("city"),
            state_prov=tba_event.get("state_prov"),
            country=tba_event.get("country"),
            website=tba_event.get("website"),
        )


match_red_teams = Table(
    "match_red_teams",
    Base.metadata,
    Column("match_id", Integer, ForeignKey("matches.id", ondelete="CASCADE"), primary_key=True),
    Column("team_id", Integer, ForeignKey("teams.id", ondelete="CASCADE"), primary_key=True),
    Column("position", Integer, nullable=False, default=1),
)

match_blue_teams = Table(
    "match_blue_teams",
    Base.metadata,
    Column("match_id", Integer, ForeignKey("matches.id", ondelete="CASCADE"), primary_key=True),
    Column("team_id", Integer, ForeignKey("teams.id", ondelete="CASCADE"), primary_key=True),
    Column("position", Integer, nullable=False, default=1),
)


class Match(Base):
    """Single FRC match (qualification or playoff)."""

    __tablename__ = "matches"
    __table_args__ = (UniqueConstraint("match_key", name="uq_match_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("events.id", ondelete="CASCADE"), nullable=False
    )
    comp_level: Mapped[str] = mapped_column(String(8), nullable=False)
    set_number: Mapped[int] = mapped_column(Integer, default=0)
    match_number: Mapped[int] = mapped_column(Integer, nullable=False)
    winning_alliance: Mapped[str | None] = mapped_column(String(8))
    red_score: Mapped[int | None] = mapped_column(Integer)
    blue_score: Mapped[int | None] = mapped_column(Integer)
    match_time: Mapped[int | None] = mapped_column(Integer)
    youtube_keys: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    event: Mapped[Event] = relationship("Event", back_populates="matches")
    red_alliance: Mapped[list[Team]] = relationship(
        "Team", secondary=match_red_teams, back_populates="red_matches"
    )
    blue_alliance: Mapped[list[Team]] = relationship(
        "Team", secondary=match_blue_teams, back_populates="blue_matches"
    )

    def __repr__(self) -> str:
        return f"<Match {self.match_key} {self.comp_level}{self.match_number}>"

    @classmethod
    def from_tba(cls, tba_match: dict, event_id: int) -> Match:
        """Build a Match instance from a TBA /match/{key} response."""
        videos = tba_match.get("videos") or []
        youtube_keys = ",".join(v.get("key", "") for v in videos if v.get("key"))
        return cls(
            match_key=tba_match["key"],
            event_id=event_id,
            comp_level=tba_match.get("comp_level", ""),
            set_number=tba_match.get("set_number", 0),
            match_number=tba_match.get("match_number", 0),
            winning_alliance=tba_match.get("winning_alliance"),
            red_score=tba_match.get("red_score"),
            blue_score=tba_match.get("blue_score"),
            match_time=tba_match.get("time"),
            youtube_keys=youtube_keys or None,
        )


def create_session_factory(database_url: str):
    """Create a SQLAlchemy engine and session factory from a connection URL."""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine), engine
