# compact_gym Documentation Index

Quick reference for all documentation files.

## 📚 Start Here

### Essential Reading (New Users & AI)

1. **[overview/ARCHITECTURE.md](overview/ARCHITECTURE.md)** ⭐
   - Complete system architecture
   - Clean layer separation (teleop → env → hardware)
   - Input agnosticism verification
   - Data flow diagrams
   - **READ THIS FIRST**

2. **[overview/ACTION_SPACE_NOTES.md](overview/ACTION_SPACE_NOTES.md)**
   - Normalized action space [-1, 1]
   - Denormalization semantics
   - Gripper action format
   - Common mistakes

## 🔧 Active Documentation

- **[../README.md](../README.md)** - Project overview and quick start (root)
- **[RUNTIME_FIXES.md](RUNTIME_FIXES.md)** - Production issues and fixes
- **[TESTING.md](TESTING.md)** - Current testing guide (validation, teleoperation)

## 📦 Archived (Historical Context)

### Phase 1: Encoder Polling
- `archived/PHASE1_COMPLETE.md` - Summary
- `archived/ENCODER_POLLING_IMPLEMENTATION.md` - Details

### Phase 2: ArUco Background Thread
- `archived/PHASE2_COMPLETE.md` - Summary
- `archived/ARUCO_THREAD_IMPLEMENTATION.md` - Details

### Phase 3: Data Collection Prep
- `archived/PHASE3_PREP_FIXES.md` - Pre-launch fixes
- `archived/PHASE3_TESTING.md` - Testing plan
- `archived/TESTING.md` - Phase 1/2 verification scripts

### Obsolete
- `archived/IMPLEMENTATION_REVIEW.md` - Initial review (pre-fixes)
- `archived/SHUTDOWN_FIX.md` - Test script fix (minor)

---

## Quick Navigation by Topic

### Architecture & Design
- [ARCHITECTURE.md](overview/ARCHITECTURE.md) - Complete architecture
- [ACTION_SPACE_NOTES.md](overview/ACTION_SPACE_NOTES.md) - Action space design

### Troubleshooting
- [RUNTIME_FIXES.md](RUNTIME_FIXES.md) - Known issues & solutions
- [TESTING.md](TESTING.md) - Testing procedures

### Implementation History
- [archived/PHASE1_COMPLETE.md](archived/PHASE1_COMPLETE.md) - Encoder polling
- [archived/PHASE2_COMPLETE.md](archived/PHASE2_COMPLETE.md) - ArUco threading
- [archived/PHASE3_PREP_FIXES.md](archived/PHASE3_PREP_FIXES.md) - Pre-production fixes

---

**For AI Assistants**: Start with [ARCHITECTURE.md](overview/ARCHITECTURE.md), then check [RUNTIME_FIXES.md](RUNTIME_FIXES.md) for current issues.
