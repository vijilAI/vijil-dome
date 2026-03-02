"""Tests for shadow mode (enforce flag) behavior."""
import pytest
from vijil_dome.Dome import ScanResult

# Default values for required ScanResult fields not under test
_DEFAULTS = dict(response_string="", trace={}, exec_time=0.0)


class TestScanResultEnforced:
    """Verify enforced field semantics across all mode combinations."""

    def test_not_flagged_not_enforced(self):
        """Safe content in any mode: flagged=False, enforced=False."""
        r = ScanResult(flagged=False, enforced=False, **_DEFAULTS)
        assert r.is_safe()
        assert not r.enforced

    def test_flagged_shadow_mode(self):
        """Flagged content in shadow mode: flagged=True, enforced=False."""
        r = ScanResult(flagged=True, enforced=False, **_DEFAULTS)
        assert not r.is_safe()  # is_safe() reflects detection, not action
        assert not r.enforced   # shadow mode: don't block

    def test_flagged_enforce_mode(self):
        """Flagged content in enforce mode: flagged=True, enforced=True."""
        r = ScanResult(flagged=True, enforced=True, **_DEFAULTS)
        assert not r.is_safe()
        assert r.enforced  # enforce mode: block

    def test_enforced_requires_flagged(self):
        """enforced=True without flagged=True is a logic error at Dome level.
        ScanResult itself allows it (Pydantic doesn't validate cross-field),
        but Dome.guard_input() should never produce this combination."""
        r = ScanResult(flagged=False, enforced=True, **_DEFAULTS)
        assert r.is_safe()  # is_safe() only checks flagged
        # This test documents the invariant: Dome sets enforced = flagged AND enforce

    def test_default_enforced_is_false(self):
        """New ScanResult defaults to enforced=False (backwards compatible)."""
        r = ScanResult(flagged=False, **_DEFAULTS)
        assert not r.enforced

    def test_is_safe_unchanged(self):
        """is_safe() returns not self.flagged, regardless of enforced."""
        assert ScanResult(flagged=False, enforced=True, **_DEFAULTS).is_safe()
        assert not ScanResult(flagged=True, enforced=False, **_DEFAULTS).is_safe()


class TestDomeEnforceFlag:
    """Verify Dome passes enforce flag to ScanResult."""

    def test_dome_default_enforce_true(self):
        from vijil_dome import Dome
        d = Dome()
        assert d.enforce is True

    def test_dome_shadow_mode(self):
        from vijil_dome import Dome
        d = Dome(enforce=False)
        assert d.enforce is False
