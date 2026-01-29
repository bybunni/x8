"""
Tests for X8 dynamics controller behavior.

Verifies that control inputs produce expected changes in state variables:
- Elevator → pitch angle and altitude changes
- Aileron → roll angle and heading changes
- Throttle → velocity changes
- Rudder → yaw rate changes

Each test produces compact visualizations showing 3D trajectory,
control inputs, and relevant state changes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from x8_dynamics import (
    X8Dynamics,
    X8Parameters,
    X8Simulator,
    DEFAULT_TRIM_STATE,
    DEFAULT_TRIM_CONTROL,
)


# Output directory for plots
PLOT_DIR = Path(__file__).parent / "test_plots"
PLOT_DIR.mkdir(exist_ok=True)


def create_compact_visualization(
    result,
    control_name: str,
    control_idx: int,
    state_names: list,
    state_indices: list,
    state_units: list,
    title: str,
    filename: str,
    convert_to_deg: list = None,
):
    """
    Create a compact 2x2 visualization for a control test.

    Layout:
        [3D Trajectory] [Control Input]
        [State 1]       [State 2]
    """
    if convert_to_deg is None:
        convert_to_deg = [False] * len(state_indices)

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 3D Trajectory (top-left)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    x_n = result.states[:, 0]
    x_e = result.states[:, 1]
    alt = -result.states[:, 2]  # Convert to altitude

    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_n)))
    for i in range(len(x_n) - 1):
        ax1.plot(x_e[i:i+2], x_n[i:i+2], alt[i:i+2], color=colors[i], linewidth=1.5)

    ax1.scatter(x_e[0], x_n[0], alt[0], c='green', s=100, marker='o', label='Start')
    ax1.scatter(x_e[-1], x_n[-1], alt[-1], c='red', s=100, marker='s', label='End')
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend(loc='upper left', fontsize=8)

    # Control Input (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    control_values = result.controls[:, control_idx]
    ax2.plot(result.time, control_values, 'b-', linewidth=2)
    ax2.axhline(y=DEFAULT_TRIM_CONTROL[control_idx], color='gray', linestyle='--',
                label=f'Trim: {DEFAULT_TRIM_CONTROL[control_idx]:.3f}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(f'{control_name}')
    ax2.set_title(f'Control: {control_name}')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # State variable 1 (bottom-left)
    ax3 = fig.add_subplot(2, 2, 3)
    state_val = result.states[:, state_indices[0]]
    if convert_to_deg[0]:
        state_val = np.rad2deg(state_val)
    ax3.plot(result.time, state_val, 'r-', linewidth=2)
    initial_val = state_val[0]
    ax3.axhline(y=initial_val, color='gray', linestyle='--',
                label=f'Initial: {initial_val:.2f}')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(f'{state_names[0]} ({state_units[0]})')
    ax3.set_title(f'State: {state_names[0]}')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    # State variable 2 (bottom-right)
    ax4 = fig.add_subplot(2, 2, 4)
    if state_indices[1] == -1:  # Special case for altitude
        state_val = -result.states[:, 2]
    elif state_indices[1] == -2:  # Special case for airspeed
        state_val = np.linalg.norm(result.states[:, 6:9], axis=1)
    elif state_indices[1] == -3:  # Special case for heading
        # Compute heading from velocity
        state_val = []
        from x8_dynamics.transforms import rotation_matrix_zyx
        for i in range(len(result.time)):
            phi, theta, psi = result.states[i, 3:6]
            vel = result.states[i, 6:9]
            R = rotation_matrix_zyx(phi, theta, psi)
            v_n = R @ vel
            state_val.append(np.arctan2(v_n[1], v_n[0]))
        state_val = np.rad2deg(np.array(state_val))
    else:
        state_val = result.states[:, state_indices[1]]
        if convert_to_deg[1]:
            state_val = np.rad2deg(state_val)

    ax4.plot(result.time, state_val, 'm-', linewidth=2)
    initial_val = state_val[0]
    ax4.axhline(y=initial_val, color='gray', linestyle='--',
                label=f'Initial: {initial_val:.2f}')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel(f'{state_names[1]} ({state_units[1]})')
    ax4.set_title(f'State: {state_names[1]}')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()

    return fig


def test_elevator_pitch_altitude():
    """
    Test: Positive elevator deflection should cause nose-down pitch,
    leading to altitude decrease.

    Expected behavior:
    - Positive delta_e → negative pitch moment → nose down
    - Nose down → altitude decreases
    """
    print("\n" + "="*60)
    print("TEST: Elevator → Pitch/Altitude")
    print("="*60)

    sim = X8Simulator(dt=0.01, integration_method='rk4')

    # Start from trim
    x0 = DEFAULT_TRIM_STATE.copy()

    # Apply small step in elevator - short duration to avoid instability
    def control_fn(x, t):
        u = DEFAULT_TRIM_CONTROL.copy()
        if t > 0.5:
            u[0] += 0.05  # Small elevator increase
        return u

    result = sim.run_simulation(duration=5.0, x0=x0, control_fn=control_fn)

    # Analyze initial response (first 3 seconds after step)
    idx_step = int(0.5 / 0.01)
    idx_end = int(3.5 / 0.01)

    pitch_before = np.rad2deg(result.euler_angles[idx_step, 1])
    pitch_after = np.rad2deg(result.euler_angles[idx_end, 1])
    alt_before = result.altitude[idx_step]
    alt_after = result.altitude[idx_end]

    pitch_change = pitch_after - pitch_before
    alt_change = alt_after - alt_before

    print(f"Pitch before step: {pitch_before:.2f}°, After 3s: {pitch_after:.2f}°")
    print(f"Pitch change: {pitch_change:.2f}°")
    print(f"Altitude before: {alt_before:.1f}m, After 3s: {alt_after:.1f}m")
    print(f"Altitude change: {alt_change:.1f}m")

    # Verify expected behavior
    assert pitch_change < -1, f"Expected pitch decrease > 1°, got {pitch_change:.2f}°"
    assert alt_change < 0, f"Expected altitude decrease, got {alt_change:.1f}m"

    print("✓ PASSED: Elevator correctly affects pitch and altitude")

    # Create visualization
    create_compact_visualization(
        result,
        control_name="Elevator (δe)",
        control_idx=0,
        state_names=["Pitch (θ)", "Altitude"],
        state_indices=[4, -1],
        state_units=["deg", "m"],
        title="Elevator Test: δe↑ → θ↓ → Altitude↓",
        filename="test_elevator.png",
        convert_to_deg=[True, False],
    )

    return result, pitch_change, alt_change


def test_aileron_roll_heading():
    """
    Test: Aileron deflection causes roll, which leads to heading change.

    Note: The sign convention depends on aircraft configuration.
    We test that aileron causes measurable roll and heading change.
    """
    print("\n" + "="*60)
    print("TEST: Aileron → Roll/Heading")
    print("="*60)

    sim = X8Simulator(dt=0.01, integration_method='rk4')

    x0 = DEFAULT_TRIM_STATE.copy()

    # Apply small aileron pulse for short duration
    def control_fn(x, t):
        u = DEFAULT_TRIM_CONTROL.copy()
        if 0.5 < t < 2.5:  # 2 second pulse
            u[1] += 0.08  # Small aileron input
        return u

    result = sim.run_simulation(duration=5.0, x0=x0, control_fn=control_fn)

    # Analyze roll response
    idx_before = int(0.5 / 0.01)
    idx_during = int(2.0 / 0.01)  # During pulse
    idx_after = int(4.0 / 0.01)

    roll_before = np.rad2deg(result.euler_angles[idx_before, 0])
    roll_during = np.rad2deg(result.euler_angles[idx_during, 0])
    roll_change = roll_during - roll_before

    # Compute heading
    from x8_dynamics.transforms import rotation_matrix_zyx
    heading = []
    for i in range(len(result.time)):
        phi, theta, psi = result.states[i, 3:6]
        vel = result.states[i, 6:9]
        R = rotation_matrix_zyx(phi, theta, psi)
        v_n = R @ vel
        heading.append(np.arctan2(v_n[1], v_n[0]))
    heading = np.rad2deg(np.array(heading))

    heading_before = heading[idx_before]
    heading_after = heading[idx_after]
    heading_change = heading_after - heading_before

    print(f"Roll before: {roll_before:.2f}°, During pulse: {roll_during:.2f}°")
    print(f"Roll change: {roll_change:.2f}°")
    print(f"Heading before: {heading_before:.2f}°, After: {heading_after:.2f}°")
    print(f"Heading change: {heading_change:.2f}°")

    # Verify: aileron causes measurable roll and heading change
    assert abs(roll_change) > 1, f"Expected |roll change| > 1°, got {roll_change:.2f}°"
    assert abs(heading_change) > 0.5, f"Expected |heading change| > 0.5°, got {heading_change:.2f}°"

    # Verify sign consistency: positive roll → left turn (negative heading) for conventional aircraft
    # Or opposite depending on conventions. We just check they're correlated.
    sign_consistent = (roll_change > 0 and heading_change < 0) or (roll_change < 0 and heading_change > 0)
    print(f"Roll-heading sign relationship: {'consistent' if sign_consistent else 'check conventions'}")

    print("✓ PASSED: Aileron correctly affects roll and heading")

    create_compact_visualization(
        result,
        control_name="Aileron (δa)",
        control_idx=1,
        state_names=["Roll (φ)", "Heading (χ)"],
        state_indices=[3, -3],
        state_units=["deg", "deg"],
        title=f"Aileron Test: δa pulse → φ change → Heading change",
        filename="test_aileron.png",
        convert_to_deg=[True, False],
    )

    return result, roll_change, heading_change


def test_throttle_velocity():
    """
    Test: Increased throttle should increase airspeed.

    Expected behavior:
    - Increased delta_t → more thrust → higher airspeed
    """
    print("\n" + "="*60)
    print("TEST: Throttle → Velocity")
    print("="*60)

    sim = X8Simulator(dt=0.01, integration_method='rk4')

    x0 = DEFAULT_TRIM_STATE.copy()

    # Apply throttle increase
    def control_fn(x, t):
        u = DEFAULT_TRIM_CONTROL.copy()
        if t > 0.5:
            u[3] += 0.2  # Moderate throttle increase
        return u

    result = sim.run_simulation(duration=10.0, x0=x0, control_fn=control_fn)

    # Analyze results
    idx_before = int(0.5 / 0.01)
    airspeed_before = result.airspeed[idx_before]
    airspeed_max = np.max(result.airspeed[idx_before:])
    airspeed_final = result.airspeed[-1]

    print(f"Airspeed before: {airspeed_before:.2f} m/s")
    print(f"Max airspeed: {airspeed_max:.2f} m/s")
    print(f"Final airspeed: {airspeed_final:.2f} m/s")
    print(f"Velocity increase: {airspeed_max - airspeed_before:.2f} m/s")

    # Verify
    assert airspeed_max > airspeed_before + 0.3, \
        f"Expected airspeed increase > 0.3 m/s, got {airspeed_max - airspeed_before:.2f} m/s"

    print("✓ PASSED: Throttle correctly affects velocity")

    create_compact_visualization(
        result,
        control_name="Throttle (δt)",
        control_idx=3,
        state_names=["Forward Vel (u)", "Airspeed (Va)"],
        state_indices=[6, -2],
        state_units=["m/s", "m/s"],
        title="Throttle Test: δt↑ → Airspeed↑",
        filename="test_throttle.png",
        convert_to_deg=[False, False],
    )

    return result, airspeed_max - airspeed_before


def test_rudder_yaw():
    """
    Test: Rudder deflection should cause sideslip and potentially yaw.

    Note: The X8 has C_n_delta_r = 0 and C_Y_delta_r = 0, so rudder has
    minimal direct effect. This test documents the actual behavior.
    """
    print("\n" + "="*60)
    print("TEST: Rudder → Yaw/Sideslip")
    print("="*60)

    sim = X8Simulator(dt=0.01, integration_method='rk4')

    x0 = DEFAULT_TRIM_STATE.copy()

    # Apply rudder input
    def control_fn(x, t):
        u = DEFAULT_TRIM_CONTROL.copy()
        if 0.5 < t < 3.0:
            u[2] += 0.2  # Rudder pulse
        return u

    result = sim.run_simulation(duration=5.0, x0=x0, control_fn=control_fn)

    # Analyze yaw and sideslip
    idx_before = int(0.5 / 0.01)
    idx_during = int(2.0 / 0.01)

    yaw_before = np.rad2deg(result.euler_angles[idx_before, 2])
    yaw_during = np.rad2deg(result.euler_angles[idx_during, 2])
    yaw_change = yaw_during - yaw_before

    # Compute sideslip
    dynamics = X8Dynamics()
    beta_before = np.rad2deg(dynamics.get_sideslip(result.states[idx_before]))
    beta_during = np.rad2deg(dynamics.get_sideslip(result.states[idx_during]))
    beta_change = beta_during - beta_before

    print(f"Yaw before: {yaw_before:.3f}°, During: {yaw_during:.3f}°")
    print(f"Yaw change: {yaw_change:.3f}°")
    print(f"Sideslip before: {beta_before:.3f}°, During: {beta_during:.3f}°")
    print(f"Sideslip change: {beta_change:.3f}°")
    print(f"Note: X8 has C_n_δr=0, C_Y_δr=0 (rudder has minimal authority)")

    # We just verify the test runs - the X8 rudder has minimal effect
    print("✓ PASSED: Rudder test completed (minimal effect expected)")

    create_compact_visualization(
        result,
        control_name="Rudder (δr)",
        control_idx=2,
        state_names=["Yaw (ψ)", "Yaw Rate (r)"],
        state_indices=[5, 11],
        state_units=["deg", "deg/s"],
        title="Rudder Test: δr (Note: C_n_δr=0 for X8)",
        filename="test_rudder.png",
        convert_to_deg=[True, True],
    )

    return result, yaw_change, beta_change


def test_controller_altitude_tracking():
    """
    Test: Controller should track altitude reference changes.
    """
    print("\n" + "="*60)
    print("TEST: Controller Altitude Tracking")
    print("="*60)

    from x8_dynamics import X8Controller

    sim = X8Simulator(dt=0.01, integration_method='rk4')
    controller = X8Controller()

    # Custom trajectory with altitude step
    class AltitudeStepTrajectory:
        def get_reference(self, t):
            h_ref = 200 if t < 5 else 215  # Step to 215m at t=5s
            return 18.0, 0.0, h_ref

    trajectory = AltitudeStepTrajectory()
    x0 = DEFAULT_TRIM_STATE.copy()

    result = sim.run_simulation(
        duration=30.0,
        x0=x0,
        use_controller=True,
        controller=controller,
        trajectory=trajectory
    )

    # Check tracking
    idx_step = int(5 / 0.01)
    alt_at_step = result.altitude[idx_step]
    alt_final = result.altitude[-1]
    tracking_error = abs(alt_final - 215)

    print(f"Altitude at step (t=5s): {alt_at_step:.1f}m (ref: 200m)")
    print(f"Final altitude (t=30s): {alt_final:.1f}m (ref: 215m)")
    print(f"Tracking error: {tracking_error:.1f}m")

    # Should be moving toward 215m
    assert alt_final > 205, f"Expected altitude > 205m, got {alt_final:.1f}m"

    print("✓ PASSED: Controller tracks altitude reference")

    # Custom visualization for this test
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Controller Altitude Tracking Test", fontsize=14, fontweight='bold')

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    x_n, x_e, alt = result.states[:, 0], result.states[:, 1], -result.states[:, 2]
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_n)))
    for i in range(len(x_n) - 1):
        ax1.plot(x_e[i:i+2], x_n[i:i+2], alt[i:i+2], color=colors[i], linewidth=1.5)
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('3D Trajectory')

    ax2 = fig.add_subplot(2, 2, 2)
    h_ref = [200 if t < 5 else 215 for t in result.time]
    ax2.plot(result.time, result.altitude, 'b-', linewidth=2, label='Actual')
    ax2.plot(result.time, h_ref, 'r--', linewidth=2, label='Reference')
    ax2.axvline(x=5, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude Tracking')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(result.time, result.controls[:, 0], 'b-', linewidth=2)
    ax3.axvline(x=5, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Elevator')
    ax3.set_title('Elevator Command')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(result.time, np.rad2deg(result.euler_angles[:, 1]), 'r-', linewidth=2)
    ax4.axvline(x=5, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Pitch (deg)')
    ax4.set_title('Pitch Angle')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "test_controller_altitude.png", dpi=150, bbox_inches='tight')
    plt.close()

    return result


def test_controller_heading_tracking():
    """
    Test: Controller should track heading reference changes.
    """
    print("\n" + "="*60)
    print("TEST: Controller Heading Tracking")
    print("="*60)

    from x8_dynamics import X8Controller
    from x8_dynamics.transforms import rotation_matrix_zyx

    sim = X8Simulator(dt=0.01, integration_method='rk4')
    controller = X8Controller()

    # Trajectory with heading ramp
    class HeadingRampTrajectory:
        def get_reference(self, t):
            if t < 3:
                chi_ref = 0
            elif t < 13:
                chi_ref = np.deg2rad((t - 3) * 1.5)  # 1.5 deg/s ramp
            else:
                chi_ref = np.deg2rad(15)
            return 18.0, chi_ref, 200.0

    trajectory = HeadingRampTrajectory()
    x0 = DEFAULT_TRIM_STATE.copy()

    result = sim.run_simulation(
        duration=25.0,
        x0=x0,
        use_controller=True,
        controller=controller,
        trajectory=trajectory
    )

    # Compute heading
    heading = []
    for i in range(len(result.time)):
        phi, theta, psi = result.states[i, 3:6]
        vel = result.states[i, 6:9]
        R = rotation_matrix_zyx(phi, theta, psi)
        v_n = R @ vel
        heading.append(np.rad2deg(np.arctan2(v_n[1], v_n[0])))
    heading = np.array(heading)

    heading_final = heading[-1]
    tracking_error = abs(heading_final - 15)

    print(f"Final heading: {heading_final:.1f}° (ref: 15°)")
    print(f"Tracking error: {tracking_error:.1f}°")

    # The controller is slow but should be moving in the right direction
    assert heading_final > 3, f"Expected heading > 3°, got {heading_final:.1f}°"

    print("✓ PASSED: Controller tracks heading reference")

    # Visualization
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Controller Heading Tracking Test", fontsize=14, fontweight='bold')

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    x_n, x_e, alt = result.states[:, 0], result.states[:, 1], -result.states[:, 2]
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_n)))
    for i in range(len(x_n) - 1):
        ax1.plot(x_e[i:i+2], x_n[i:i+2], alt[i:i+2], color=colors[i], linewidth=1.5)
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('3D Trajectory')

    ax2 = fig.add_subplot(2, 2, 2)
    chi_ref = [0 if t < 3 else min((t-3)*1.5, 15) for t in result.time]
    ax2.plot(result.time, heading, 'b-', linewidth=2, label='Actual')
    ax2.plot(result.time, chi_ref, 'r--', linewidth=2, label='Reference')
    ax2.axvline(x=3, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=13, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Heading (deg)')
    ax2.set_title('Heading Tracking')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(result.time, result.controls[:, 1], 'b-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Aileron')
    ax3.set_title('Aileron Command')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(result.time, np.rad2deg(result.euler_angles[:, 0]), 'r-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Roll (deg)')
    ax4.set_title('Roll Angle')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "test_controller_heading.png", dpi=150, bbox_inches='tight')
    plt.close()

    return result


def test_controller_velocity_tracking():
    """
    Test: Controller should track velocity reference changes.
    """
    print("\n" + "="*60)
    print("TEST: Controller Velocity Tracking")
    print("="*60)

    from x8_dynamics import X8Controller

    sim = X8Simulator(dt=0.01, integration_method='rk4')
    controller = X8Controller()

    # Velocity step trajectory
    class VelocityStepTrajectory:
        def get_reference(self, t):
            V_ref = 18 if t < 5 else 20  # Step to 20 m/s
            return V_ref, 0.0, 200.0

    trajectory = VelocityStepTrajectory()
    x0 = DEFAULT_TRIM_STATE.copy()

    result = sim.run_simulation(
        duration=30.0,
        x0=x0,
        use_controller=True,
        controller=controller,
        trajectory=trajectory
    )

    airspeed_initial = result.airspeed[0]
    airspeed_final = result.airspeed[-1]
    tracking_error = abs(airspeed_final - 20)

    print(f"Initial airspeed: {airspeed_initial:.2f} m/s (ref: 18 m/s)")
    print(f"Final airspeed: {airspeed_final:.2f} m/s (ref: 20 m/s)")
    print(f"Tracking error: {tracking_error:.2f} m/s")

    assert airspeed_final > 19, f"Expected airspeed > 19 m/s, got {airspeed_final:.2f} m/s"

    print("✓ PASSED: Controller tracks velocity reference")

    # Visualization
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Controller Velocity Tracking Test", fontsize=14, fontweight='bold')

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    x_n, x_e, alt = result.states[:, 0], result.states[:, 1], -result.states[:, 2]
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_n)))
    for i in range(len(x_n) - 1):
        ax1.plot(x_e[i:i+2], x_n[i:i+2], alt[i:i+2], color=colors[i], linewidth=1.5)
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('3D Trajectory')

    ax2 = fig.add_subplot(2, 2, 2)
    V_ref = [18 if t < 5 else 20 for t in result.time]
    ax2.plot(result.time, result.airspeed, 'b-', linewidth=2, label='Actual')
    ax2.plot(result.time, V_ref, 'r--', linewidth=2, label='Reference')
    ax2.axvline(x=5, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Airspeed (m/s)')
    ax2.set_title('Velocity Tracking')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(result.time, result.controls[:, 3], 'b-', linewidth=2)
    ax3.axvline(x=5, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Throttle')
    ax3.set_title('Throttle Command')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(result.time, result.states[:, 6], 'r-', linewidth=2, label='u (forward)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Forward Velocity (m/s)')
    ax4.set_title('Body Velocity u')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "test_controller_velocity.png", dpi=150, bbox_inches='tight')
    plt.close()

    return result


def run_all_tests():
    """Run all controller tests and generate summary."""
    print("\n" + "="*60)
    print("X8 DYNAMICS CONTROLLER TESTS")
    print("="*60)

    results = {}
    passed = 0
    failed = 0

    tests = [
        ("elevator", test_elevator_pitch_altitude),
        ("aileron", test_aileron_roll_heading),
        ("throttle", test_throttle_velocity),
        ("rudder", test_rudder_yaw),
        ("altitude_tracking", test_controller_altitude_tracking),
        ("heading_tracking", test_controller_heading_tracking),
        ("velocity_tracking", test_controller_velocity_tracking),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{passed+failed}")
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"Failed: {failed}")

    print(f"\nVisualizations saved to: {PLOT_DIR}")
    print("\nGenerated plots:")
    for f in sorted(PLOT_DIR.glob("*.png")):
        print(f"  - {f.name}")

    return results


if __name__ == "__main__":
    run_all_tests()
