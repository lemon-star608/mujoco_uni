from __future__ import annotations

import importlib.metadata
import re
from typing import Any

import mujoco
import pytest

import mujoco_uni


def _version_tuple(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    assert match is not None
    return tuple(int(match.group(i)) for i in range(1, 4))


def test_package_version_is_independent_from_solver_version() -> None:
    assert importlib.metadata.version("mujoco-uni-runtime") == mujoco_uni.__version__
    assert mujoco_uni.__version__ == "0.4.0.dev0"
    assert mujoco_uni.MUJOCO_DEFAULT_VERSION == "3.8.0"
    assert mujoco_uni.MUJOCO_MIN_VERSION == "3.5.0"
    assert mujoco_uni.MUJOCO_MAX_VERSION_EXCLUSIVE == "3.11.0"
    assert mujoco_uni.MUJOCO_VERSION_SPEC == ">=3.5,<3.11"
    assert (3, 5, 0) <= _version_tuple(mujoco.__version__) < (3, 11, 0)


def test_batch_env_constructs_from_official_mujoco_model() -> None:
    from mujoco_uni.batch_env import SUPPORTED_FIELDS, BatchEnvPool
    from mujoco_uni.compiled import NativeBatchEnvPool, batch_available, batch_import_error
    from mujoco_uni.runtime import available_backends, batch_diagnostics

    mj: Any = mujoco
    model = mj.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="box">
              <freejoint/>
              <geom type="box" size="0.1 0.1 0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    assert batch_available()
    assert batch_import_error() is None
    assert available_backends() == {"batch": True}
    assert batch_diagnostics()["batch_import_error"] is None
    assert mujoco_uni.BatchEnvPool is BatchEnvPool
    assert NativeBatchEnvPool is not BatchEnvPool
    assert set(SUPPORTED_FIELDS) == {
        "body_mass",
        "body_ipos",
        "body_iquat",
        "body_inertia",
        "dof_armature",
        "gravity",
        "geom_friction",
        "kp",
        "kd",
        "geom_size",
        "geom_pos",
        "mocap_pos",
    }

    with BatchEnvPool(model, nbatch=2, nthread=1) as pool:
        assert pool.nbatch == 2
        assert pool.nthread == 1
        assert pool.nstate == mj.mj_stateSize(model, mj.mjtState.mjSTATE_FULLPHYSICS)
        assert pool.get_model(0).nbody == model.nbody


def test_mocap_pos_reset_is_batched_persistent_and_indexed() -> None:
    import numpy as np

    from mujoco_uni.batch_env import BatchEnvPool

    mj: Any = mujoco
    model = mj.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <body name="table" pos="0 0 0.38" mocap="true">
            <geom name="table_box" type="box" size="1 1 0.1"/>
          </body>
        </worldbody></mujoco>
        """
    )
    nstate = mj.mj_stateSize(model, mj.mjtState.mjSTATE_FULLPHYSICS)
    state = np.zeros((3, nstate), dtype=np.float64)
    state[:, 1 : 1 + model.nq] = model.qpos0
    with BatchEnvPool(model, nbatch=3, nthread=1) as pool:
        target = np.array([[0.0, 0.0, 0.41]], dtype=np.float64)
        pool.reset([1], state[[1]], randomization={"mocap_pos": target})
        assert np.allclose(pool.get_field(0, "mocap_pos"), [0.0, 0.0, 0.38])
        assert np.allclose(pool.get_field(1, "mocap_pos"), target.reshape(-1))
        assert np.allclose(pool.get_field(2, "mocap_pos"), [0.0, 0.0, 0.38])
        assert np.allclose(pool.get_field_indexed(1, "mocap_pos", 0), target[0])
        pool.set_field_indexed(1, "mocap_pos", 0, [0.0, 0.0, 0.42])
        assert np.allclose(pool.get_field_indexed(1, "mocap_pos", 0), [0.0, 0.0, 0.42])
        pool.reset([1], state[[1]])
        stepped = pool.step(state, nstep=1)
        assert np.allclose(pool.get_field(1, "mocap_pos"), [0.0, 0.0, 0.42])
        assert stepped.shape == state.shape


def test_geom_bounds_parity_after_growth() -> None:
    """Test 1: bounds match direct compile after growing geoms via reset and set_field_indexed."""
    import numpy as np

    from mujoco_uni.batch_env import BatchEnvPool

    mj: Any = mujoco
    xml = """
    <mujoco>
      <worldbody>
        <body name="b">
          <freejoint/>
          <geom name="sph" type="sphere" size="0.05" pos="0 0 0"/>
          <geom name="cap" type="capsule" size="0.03 0.07" pos="0.5 0 0"/>
          <geom name="cyl" type="cylinder" size="0.04 0.08" pos="1 0 0"/>
          <geom name="ell" type="ellipsoid" size="0.02 0.06 0.03" pos="1.5 0 0"/>
          <geom name="box" type="box" size="0.03 0.05 0.07" pos="2 0 0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mj.MjModel.from_xml_string(xml)

    # Compile reference models at the grown sizes
    refs = [
        mj.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="sphere" size="0.11" mass="1"/></body></worldbody></mujoco>'
        ),
        mj.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="capsule" size="0.07 0.13" mass="1"/></body></worldbody></mujoco>'
        ),
        mj.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="cylinder" size="0.08 0.15" mass="1"/></body></worldbody></mujoco>'
        ),
        mj.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="ellipsoid" size="0.05 0.09 0.04" mass="1"/></body></worldbody></mujoco>'
        ),
        mj.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="box" size="0.06 0.10 0.12" mass="1"/></body></worldbody></mujoco>'
        ),
    ]

    new_sizes = [
        [0.11, 0, 0],  # sphere
        [0.07, 0.13, 0],  # capsule
        [0.08, 0.15, 0],  # cylinder
        [0.05, 0.09, 0.04],  # ellipsoid
        [0.06, 0.10, 0.12],  # box
    ]

    # Test via reset(randomization=...)
    with BatchEnvPool(model, nbatch=1, nthread=0) as pool:
        # Flatten: 5 geoms × 3 components = 15 elements, shape (1, 15) for 1 env
        flat_sizes = np.array(new_sizes).flatten()
        pool.reset(
            env_ids=[0],
            initial_state=[[0] * pool.nstate],
            randomization={"geom_size": [flat_sizes]},
        )
        m = pool.get_model(0)
        for g in range(5):
            assert abs(float(m.geom_rbound[g]) - float(refs[g].geom_rbound[0])) < 1e-12
            assert np.allclose(m.geom_aabb[g], refs[g].geom_aabb[0])

    # Test via set_field_indexed
    with BatchEnvPool(model, nbatch=1, nthread=0) as pool:
        pool.set_field_indexed(0, "geom_size", np.array([0, 1, 2, 3, 4]), np.array(new_sizes))
        m = pool.get_model(0)
        for g in range(5):
            assert abs(float(m.geom_rbound[g]) - float(refs[g].geom_rbound[0])) < 1e-12
            assert np.allclose(m.geom_aabb[g], refs[g].geom_aabb[0])


def test_geom_bounds_parity_after_shrink() -> None:
    """Test 2: bounds match direct compile after shrinking geoms to 1e-4 (UniLab use case)."""
    import numpy as np

    from mujoco_uni.batch_env import BatchEnvPool

    mj: Any = mujoco
    xml = """
    <mujoco>
      <worldbody>
        <body name="b">
          <freejoint/>
          <geom name="box1" type="box" size="0.05 0.08 0.12"/>
          <geom name="cyl1" type="cylinder" size="0.06 0.10"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mj.MjModel.from_xml_string(xml)

    # Shrink both geoms to 1e-4 (what UniLab does to unused topology geoms)
    shrunk = [[1e-4, 1e-4, 1e-4], [1e-4, 1e-4, 0]]

    with BatchEnvPool(model, nbatch=1, nthread=0) as pool:
        # Flatten: 2 geoms × 3 components = 6 elements
        flat_shrunk = np.array(shrunk).flatten()
        pool.reset(
            env_ids=[0],
            initial_state=[[0] * pool.nstate],
            randomization={"geom_size": [flat_shrunk]},
        )
        m = pool.get_model(0)

        ref_box = mj.MjModel.from_xml_string(
            "<mujoco><worldbody><body><freejoint/>"
            '<geom type="box" size="0.0001 0.0001 0.0001" mass="1"/></body></worldbody></mujoco>'
        )
        ref_cyl = mj.MjModel.from_xml_string(
            "<mujoco><worldbody><body><freejoint/>"
            '<geom type="cylinder" size="0.0001 0.0001" mass="1"/></body></worldbody></mujoco>'
        )

        assert abs(float(m.geom_rbound[0]) - float(ref_box.geom_rbound[0])) < 1e-15
        assert abs(float(m.geom_rbound[1]) - float(ref_cyl.geom_rbound[0])) < 1e-15
        assert np.allclose(m.geom_aabb[0], ref_box.geom_aabb[0])
        assert np.allclose(m.geom_aabb[1], ref_cyl.geom_aabb[0])


def test_bug_a_regression_geom_size_pos_indexed() -> None:
    """Test 3: Bug A regression — set_field_indexed for geom_size/geom_pos round-trips correctly."""
    import numpy as np

    from mujoco_uni.batch_env import BatchEnvPool

    mj: Any = mujoco
    xml = """
    <mujoco>
      <worldbody>
        <body><freejoint/><geom type="box" size="0.05 0.08 0.12" pos="1 2 3"/></body>
      </worldbody>
    </mujoco>
    """
    model = mj.MjModel.from_xml_string(xml)

    with BatchEnvPool(model, nbatch=1, nthread=0) as pool:
        # Test geom_size
        pool.set_field_indexed(0, "geom_size", np.array([0]), np.array([[0.06, 0.09, 0.13]]))
        assert np.allclose(pool.get_field(0, "geom_size"), [0.06, 0.09, 0.13])
        assert np.allclose(
            pool.get_field_indexed(0, "geom_size", np.array([0])), [[0.06, 0.09, 0.13]]
        )
        assert np.allclose(pool.get_model(0).geom_size[0], [0.06, 0.09, 0.13])

        # Test geom_pos
        pool.set_field_indexed(0, "geom_pos", np.array([0]), np.array([[4, 5, 6]]))
        assert np.allclose(pool.get_field(0, "geom_pos"), [4, 5, 6])
        assert np.allclose(pool.get_field_indexed(0, "geom_pos", np.array([0])), [[4, 5, 6]])
        assert np.allclose(pool.get_model(0).geom_pos[0], [4, 5, 6])


def test_physics_regression_ncon_after_growth() -> None:
    """Test 4: bounds match direct compile after growing box - validates geom bounds refresh."""
    import numpy as np

    from mujoco_uni.batch_env import BatchEnvPool

    mj: Any = mujoco
    xml_template = """
    <mujoco>
      <worldbody>
        <geom name="table" type="box" size="1 1 0.05" pos="0 0 0.05"/>
        <body name="obj" pos="0 0 0.145">
          <freejoint/>
          <geom name="obj_geom" type="box" size="%s %s %s"/>
        </body>
      </worldbody>
    </mujoco>
    """

    # Compile at small size 0.01, then grow to 0.06
    model_small = mj.MjModel.from_xml_string(xml_template % ("0.01", "0.01", "0.01"))
    model_large = mj.MjModel.from_xml_string(xml_template % ("0.06", "0.06", "0.06"))

    with BatchEnvPool(model_small, nbatch=1, nthread=0) as pool:
        # 2 geoms (table + obj), flatten to (1, 6)
        flat_sizes = np.array([[0.01, 0.01, 0.01], [0.06, 0.06, 0.06]]).flatten()
        pool.reset(
            env_ids=[0],
            initial_state=[[0] * pool.nstate],
            randomization={"geom_size": [flat_sizes]},
        )
        m = pool.get_model(0)

        # Verify bounds match direct compile (the core fix for Bug B)
        assert np.allclose(m.geom_size[1], model_large.geom_size[1])
        assert abs(float(m.geom_rbound[1]) - float(model_large.geom_rbound[1])) < 1e-12
        assert np.allclose(m.geom_aabb[1], model_large.geom_aabb[1])


def test_three_geom_same_body_topology() -> None:
    """Test 5: three-geom same-body topology mirrors handle_head_base.xml shrink-and-bury."""
    import numpy as np

    from mujoco_uni.batch_env import BatchEnvPool

    mj: Any = mujoco
    xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 0"/>
        <body pos="0 0 0.2">
          <freejoint/>
          <geom name="g1" type="box" size="0.05 0.05 0.05"/>
          <geom name="g2" type="cylinder" size="0.04 0.08"/>
          <geom name="g3" type="box" size="0.03 0.03 0.03"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mj.MjModel.from_xml_string(xml)

    # Shrink g1 to 1e-4 and leave g2, g3 at original size
    sizes = [
        [0, 0, 0],  # floor (plane, ignored)
        [1e-4, 1e-4, 1e-4],  # g1 buried
        [0.04, 0.08, 0],  # g2 active
        [0.03, 0.03, 0.03],  # g3 active
    ]

    with BatchEnvPool(model, nbatch=1, nthread=0) as pool:
        # 4 geoms × 3 components = 12 elements, flatten for 1 env
        flat_sizes = np.array(sizes).flatten()
        pool.reset(
            env_ids=[0],
            initial_state=[[0] * pool.nstate],
            randomization={"geom_size": [flat_sizes]},
        )
        m = pool.get_model(0)

        # Check g1 bounds match direct compile at 1e-4
        ref_buried = mj.MjModel.from_xml_string(
            "<mujoco><worldbody><body><freejoint/>"
            '<geom type="box" size="0.0001 0.0001 0.0001" mass="1"/></body></worldbody></mujoco>'
        )
        assert abs(float(m.geom_rbound[1]) - float(ref_buried.geom_rbound[0])) < 1e-15

        # Check g2, g3 bounds unchanged
        assert abs(float(m.geom_rbound[2]) - float(model.geom_rbound[2])) < 1e-12
        assert abs(float(m.geom_rbound[3]) - float(model.geom_rbound[3])) < 1e-12

        # Run physics: buried geom should produce no floor contact
        d = mj.MjData(m)
        mj.mj_step(m, d)
        # g2 and g3 should contact floor (geom 0), but g1 should not
        g1_contacts = sum(
            1 for i in range(d.ncon) if d.contact[i].geom1 == 1 or d.contact[i].geom2 == 1
        )
        assert g1_contacts == 0, (
            f"Buried geom g1 should not contact floor, got {g1_contacts} contacts"
        )


def test_non_geom_size_fields_leave_bounds_alone() -> None:
    """Test 6: writing geom_pos, body_mass, geom_size on a mesh leaves mesh bounds untouched."""
    import numpy as np

    from mujoco_uni.batch_env import BatchEnvPool

    mj: Any = mujoco
    xml = """
    <mujoco>
      <asset>
        <mesh name="m" vertex="0 0 0  0.3 0 0  0 0.2 0  0 0 0.15"/>
      </asset>
      <worldbody>
        <body name="b">
          <freejoint/>
          <geom name="box_g" type="box" size="0.05 0.05 0.05" pos="0 0 0"/>
          <geom name="mesh_g" type="mesh" mesh="m" pos="0.5 0 0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mj.MjModel.from_xml_string(xml)

    with BatchEnvPool(model, nbatch=1, nthread=0) as pool:
        orig_box_rb = float(model.geom_rbound[0])
        orig_box_aabb = model.geom_aabb[0].copy()
        orig_mesh_rb = float(model.geom_rbound[1])
        orig_mesh_aabb = model.geom_aabb[1].copy()

        # Write geom_pos (should not change bounds)
        pool.set_field_indexed(0, "geom_pos", np.array([0, 1]), np.array([[1, 2, 3], [4, 5, 6]]))
        m1 = pool.get_model(0)
        assert abs(float(m1.geom_rbound[0]) - orig_box_rb) < 1e-15
        assert abs(float(m1.geom_rbound[1]) - orig_mesh_rb) < 1e-15
        assert np.allclose(m1.geom_aabb[0], orig_box_aabb)
        assert np.allclose(m1.geom_aabb[1], orig_mesh_aabb)

        # Write body_mass (should not change bounds)
        # Model has 2 bodies: world (id 0) and the freejoint body (id 1)
        pool.reset(
            env_ids=[0], initial_state=[[0] * pool.nstate], randomization={"body_mass": [[1, 5]]}
        )
        m2 = pool.get_model(0)
        assert abs(float(m2.geom_rbound[0]) - orig_box_rb) < 1e-15
        assert abs(float(m2.geom_rbound[1]) - orig_mesh_rb) < 1e-15

        # Write geom_size on the mesh (should leave mesh bounds alone, box should change)
        pool.set_field_indexed(0, "geom_size", np.array([1]), np.array([[0.1, 0.2, 0.3]]))
        m3 = pool.get_model(0)
        assert abs(float(m3.geom_rbound[1]) - orig_mesh_rb) < 1e-15
        assert np.allclose(m3.geom_aabb[1], orig_mesh_aabb)


def test_per_env_isolation_geom_size_and_bounds() -> None:
    """Test 7: writing env 0 does not change env 1's geom_size or bounds."""
    import numpy as np

    from mujoco_uni.batch_env import BatchEnvPool

    mj: Any = mujoco
    xml = """
    <mujoco>
      <worldbody>
        <body><freejoint/><geom type="box" size="0.05 0.08 0.12"/></body>
      </worldbody>
    </mujoco>
    """
    model = mj.MjModel.from_xml_string(xml)

    with BatchEnvPool(model, nbatch=2, nthread=0) as pool:
        orig_size = model.geom_size[0].copy()
        orig_rb = float(model.geom_rbound[0])
        orig_aabb = model.geom_aabb[0].copy()

        # Write env 0
        pool.set_field_indexed(0, "geom_size", np.array([0]), np.array([[0.10, 0.15, 0.20]]))

        # Check env 1 unchanged
        m1 = pool.get_model(1)
        assert np.allclose(m1.geom_size[0], orig_size)
        assert abs(float(m1.geom_rbound[0]) - orig_rb) < 1e-15
        assert np.allclose(m1.geom_aabb[0], orig_aabb)


def test_indexed_geom_size_write_refreshes_rbound() -> None:
    """Indexed geom_size write must leave geom_rbound consistent with the size.

    Locks the WriteIndexedField / WriteField asymmetry: WriteField's caller
    follows up with RefreshAllGeomBounds, WriteIndexedField refreshes the single
    geom itself. A 0.14 half-extent box has rbound sqrt(3) * 0.14 == 0.242487.
    """
    import numpy as np

    from mujoco_uni.batch_env import BatchEnvPool

    mj: Any = mujoco
    xml = """
    <mujoco>
      <worldbody>
        <body><freejoint/><geom name="g" type="box" size="0.05 0.05 0.05"/></body>
      </worldbody>
    </mujoco>
    """
    model = mj.MjModel.from_xml_string(xml)
    expected_rbound = float(np.sqrt(3.0) * 0.14)
    assert abs(expected_rbound - 0.242487) < 1e-6

    with BatchEnvPool(model, nbatch=2, nthread=0) as pool:
        pool.set_field_indexed(0, "geom_size", np.array([0]), np.array([[0.14, 0.14, 0.14]]))
        m0 = pool.get_model(0)
        assert np.allclose(m0.geom_size[0], [0.14, 0.14, 0.14])
        assert abs(float(m0.geom_rbound[0]) - expected_rbound) < 1e-12
        assert np.allclose(m0.geom_aabb[0], [0, 0, 0, 0.14, 0.14, 0.14])

        # Untouched neighbour keeps the compiled bounds.
        m1 = pool.get_model(1)
        assert abs(float(m1.geom_rbound[0]) - float(model.geom_rbound[0])) < 1e-15


# ---------------------------------------------------------------------------
# Autoreset / warning observability.
#
# MuJoCo's mj_checkPos / mj_checkVel / mj_checkAcc do not merely log: they call
# mj_resetData, zeroing qpos / qvel / ctrl mid-step. The step kernel detects the
# warning, aborts its remaining substeps, and writes that post-reset state out
# as an ordinary result. These tests pin the per-env flag that makes it visible.
# ---------------------------------------------------------------------------

# Finite ctrl through a huge gear on a near-massless link: qacc overflows on the
# first mj_step, so mjWARN_BADQACC fires deterministically.
_BLOWUP_XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="arm">
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.005 0.05" fromto="0 0 0 0.1 0 0" mass="1e-6"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m0" joint="j0" gear="1e9" ctrlrange="-1e9 1e9"/>
  </actuator>
</mujoco>
"""


def test_warning_surface_constants() -> None:
    from mujoco_uni.batch_env import AUTORESET_WARNINGS, NO_WARNING, WARNING_NAMES

    mj: Any = mujoco
    assert NO_WARNING == -1
    assert len(WARNING_NAMES) == int(mj.mjtWarning.mjNWARNING)
    assert WARNING_NAMES[int(mj.mjtWarning.mjWARN_BADQACC)] == "BADQACC"
    assert WARNING_NAMES[int(mj.mjtWarning.mjWARN_BADQVEL)] == "BADQVEL"
    # Exactly the three checks that call mj_resetData.
    assert set(AUTORESET_WARNINGS) == {
        int(mj.mjtWarning.mjWARN_BADQPOS),
        int(mj.mjtWarning.mjWARN_BADQVEL),
        int(mj.mjtWarning.mjWARN_BADQACC),
    }


@pytest.mark.parametrize("nbatch", [1, 4])
@pytest.mark.parametrize("nthread", [0, 2])
def test_warning_surface_is_readable_before_first_step(nbatch: int, nthread: int) -> None:
    """Every accessor must be safe to read on a pool that has run nothing.

    The vectors stay empty until their primitive runs once, so the shape and
    the fill both come from the ``nbatch_`` fallback in ``WarningArray``
    rather than from stored data. A build where that fallback is skipped reads
    an empty vector at ``nbatch`` width and segfaults -- and the caller sees a
    crash nowhere near the cause. Training reads this on the first ``reset``
    before any ``step``, so the fallback is contractual, not defensive.

    The dtype and the shape are the load-bearing assertions: a degenerate
    zero-length return would satisfy an ``np.all(... == NO_WARNING)`` check
    vacuously.
    """
    import numpy as np

    from mujoco_uni.batch_env import NO_WARNING, BatchEnvPool, warning_is_autoreset

    mj: Any = mujoco
    model = mj.MjModel.from_xml_string(_BLOWUP_XML)

    with BatchEnvPool(model, nbatch=nbatch, nthread=nthread) as pool:
        # No step / forward / reset call before this point, by construction.
        for name in ("last_step_warning", "last_forward_warning", "last_reset_warning"):
            warning = getattr(pool, name)
            assert warning.shape == (nbatch,), (name, warning)
            assert warning.dtype == np.int32, (name, warning.dtype)
            assert warning.tolist() == [NO_WARNING] * nbatch, (name, warning)
            assert not warning_is_autoreset(warning).any(), name

        autoreset = pool.was_autoreset
        assert autoreset.shape == (nbatch,), autoreset
        assert autoreset.dtype == np.bool_, autoreset.dtype
        assert autoreset.tolist() == [False] * nbatch, autoreset


def test_step_reports_no_warning_on_healthy_rollout() -> None:
    import numpy as np

    from mujoco_uni.batch_env import NO_WARNING, BatchEnvPool

    mj: Any = mujoco
    model = mj.MjModel.from_xml_string(_BLOWUP_XML)

    for nthread in (0, 2):
        with BatchEnvPool(model, nbatch=4, nthread=nthread) as pool:
            # A pool that has never stepped reports clean, not garbage.
            assert pool.last_step_warning.shape == (4,)
            assert np.all(pool.last_step_warning == NO_WARNING)
            assert not pool.was_autoreset.any()

            state = np.zeros((4, pool.nstate), dtype=np.float64)
            state[:, 1] = 0.1  # qpos
            ctrl = np.zeros((4, model.nu), dtype=np.float64)
            pool.step(state, nstep=5, control=ctrl, return_sensor=True)

            assert np.all(pool.last_step_warning == NO_WARNING)
            assert pool.was_autoreset.dtype == np.bool_
            assert not pool.was_autoreset.any()


def test_step_flags_autoreset_per_env() -> None:
    """The flag must isolate the blown-up env from its healthy neighbours."""
    import numpy as np

    from mujoco_uni.batch_env import NO_WARNING, BatchEnvPool

    mj: Any = mujoco
    model = mj.MjModel.from_xml_string(_BLOWUP_XML)
    idx_qpos = 1

    for nthread in (0, 2):
        with BatchEnvPool(model, nbatch=4, nthread=nthread) as pool:
            state = np.zeros((4, pool.nstate), dtype=np.float64)
            state[:, idx_qpos] = 0.1
            ctrl = np.zeros((4, model.nu), dtype=np.float64)
            ctrl[2, 0] = 1e9  # only env 2

            out_state, _ = pool.step(state, nstep=3, control=ctrl, return_sensor=True)

            warning = pool.last_step_warning
            assert warning.tolist() == [
                NO_WARNING,
                NO_WARNING,
                int(mj.mjtWarning.mjWARN_BADQACC),
                NO_WARNING,
            ], warning
            assert pool.was_autoreset.tolist() == [False, False, True, False]

            # The returned state for env 2 is the post-mj_resetData state: this
            # is exactly the teleport the flag exists to expose.
            assert out_state[2, idx_qpos] == 0.0
            for env in (0, 1, 3):
                assert abs(out_state[env, idx_qpos] - 0.1) < 1e-9

            # The flag is per-call, not sticky.
            pool.step(state, nstep=3, control=np.zeros_like(ctrl), return_sensor=True)
            assert np.all(pool.last_step_warning == NO_WARNING)
            assert not pool.was_autoreset.any()


def test_step_flags_autoreset_for_multiple_envs() -> None:
    import numpy as np

    from mujoco_uni.batch_env import BatchEnvPool

    mj: Any = mujoco
    model = mj.MjModel.from_xml_string(_BLOWUP_XML)

    with BatchEnvPool(model, nbatch=8, nthread=2) as pool:
        state = np.zeros((8, pool.nstate), dtype=np.float64)
        state[:, 1] = 0.1
        ctrl = np.zeros((8, model.nu), dtype=np.float64)
        bad = [1, 4, 7]
        ctrl[bad, 0] = 1e9

        pool.step(state, nstep=2, control=ctrl, return_sensor=True)
        expected = np.zeros((8,), dtype=bool)
        expected[bad] = True
        assert pool.was_autoreset.tolist() == expected.tolist()


def test_step_return_shape_is_unchanged_by_warning_surface() -> None:
    """Backward compatibility: consumers branching on return arity still work."""
    import numpy as np

    from mujoco_uni.batch_env import BatchEnvPool

    mj: Any = mujoco
    model = mj.MjModel.from_xml_string(_BLOWUP_XML)

    with BatchEnvPool(model, nbatch=2, nthread=0) as pool:
        state = np.zeros((2, pool.nstate), dtype=np.float64)
        state[:, 1] = 0.1

        only_state = pool.step(state, nstep=1)
        assert isinstance(only_state, np.ndarray)
        assert only_state.shape == (2, pool.nstate)

        pair = pool.step(state, nstep=1, return_sensor=True)
        assert isinstance(pair, tuple) and len(pair) == 2
        assert pair[0].shape == (2, pool.nstate)
        assert pair[1].shape == (2, pool.nsensordata)


def test_forward_and_reset_warning_surface_exists_and_does_not_autoreset() -> None:
    """reset / forward run mj_forwardSkip, which never calls mj_check*.

    A bad caller-supplied qvel therefore passes through untouched on those
    paths. Pinning this keeps a future reader from assuming the step-path
    semantics apply here.
    """
    import numpy as np

    from mujoco_uni.batch_env import NO_WARNING, BatchEnvPool, warning_is_autoreset

    mj: Any = mujoco
    model = mj.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.01"/>
          <worldbody>
            <body name="arm">
              <joint name="j0" type="hinge" axis="0 0 1"/>
              <geom type="capsule" size="0.005 0.05" fromto="0 0 0 0.1 0 0" mass="0.01"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    idx_qpos, idx_qvel = 1, 1 + model.nq

    with BatchEnvPool(model, nbatch=3, nthread=0) as pool:
        assert np.all(pool.last_reset_warning == NO_WARNING)
        assert np.all(pool.last_forward_warning == NO_WARNING)

        state = np.zeros((2, pool.nstate), dtype=np.float64)
        state[:, idx_qpos] = 1.234
        state[1, idx_qvel] = 1e12  # far above mjMAXVAL

        reset_state, _ = pool.reset(env_ids=[0, 2], initial_state=state)
        # No autoreset: the absurd qvel survives mj_forwardSkip untouched.
        assert reset_state[1, idx_qvel] == 1e12
        assert not warning_is_autoreset(pool.last_reset_warning).any()
        # Env-indexed, so the env this reset skipped stays clean.
        assert pool.last_reset_warning.shape == (3,)
        assert pool.last_reset_warning[1] == NO_WARNING

        fwd_state = np.zeros((3, pool.nstate), dtype=np.float64)
        fwd_state[:, idx_qpos] = 1.234
        fwd_state[2, idx_qvel] = 1e12
        pool.forward(fwd_state)
        assert not warning_is_autoreset(pool.last_forward_warning).any()

        # Contrast: the same state through step does autoreset.
        step_state, _ = pool.step(fwd_state, nstep=1, return_sensor=True)
        assert pool.was_autoreset.tolist() == [False, False, True]
        assert step_state[2, idx_qpos] == 0.0
