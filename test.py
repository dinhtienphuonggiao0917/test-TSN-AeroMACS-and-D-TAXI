from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from math import erfc, log10, sqrt
from pathlib import Path
import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

plt.rcParams.update({
	"font.family": ["Times New Roman", "serif"],
	"font.size": 11,
	"axes.titlesize": 13,
	"axes.labelsize": 12,
	"legend.fontsize": 10,
	"xtick.labelsize": 10,
	"ytick.labelsize": 10,
	"axes.linewidth": 1.2,
	"lines.linewidth": 2.0,
	"lines.markersize": 6,
})

# Reconfigure stdout to use UTF-8 to prevent Unicode encoding errors on Windows
if hasattr(sys.stdout, "reconfigure"):
	sys.stdout.reconfigure(encoding="utf-8")

TARGET_BER = 1e-6
CHANNEL_BANDWIDTH_MHZ = 5.0
PEAK_THROUGHPUT_MBPS = 7.5
SAMPLES_PER_POINT = 2000


@dataclass(frozen=True)
class MCSConfig:
	name: str
	modulation_order: int
	code_rate: float
	repetition: int
	snr_threshold_db: float

	@property
	def bits_per_symbol(self) -> int:
		return int(log2_int(self.modulation_order))


@dataclass(frozen=True)
class ScenarioConfig:
	name: str
	mean_cinr_db: float
	k_factor_db: float
	eta: float
	shadow_sigma_db: float
	fast_fading_sigma_db: float
	deep_fade_prob: float
	deep_fade_penalty_db: float
	ar_coeff: float
	doppler_hz: float
	delay_spread_us: float


def log2_int(value: int) -> int:
	result = 0
	current = value
	while current > 1:
		current //= 2
		result += 1
	return result


def build_mcs_table() -> list[MCSConfig]:
	return [
		MCSConfig("QPSK_1_2_REP4", 4, 0.5, 4, 0.0),
		MCSConfig("QPSK_1_2_REP2", 4, 0.5, 2, 3.0),
		MCSConfig("QPSK_1_2", 4, 0.5, 1, 5.0),
		MCSConfig("16QAM_1_2", 16, 0.5, 1, 10.5),
		MCSConfig("16QAM_3_4", 16, 0.75, 1, 13.4),
		MCSConfig("64QAM_3_4", 64, 0.75, 1, 17.8),
	]


def build_scenarios(k_parking: float = -3.0, k_taxiing: float = 5.0) -> dict[str, ScenarioConfig]:
	return {
		"parking": ScenarioConfig(
			name="Parking",
			mean_cinr_db=8.0,
			k_factor_db=k_parking,
			eta=4.0,
			shadow_sigma_db=8.0,  # Adjusted to 8 dB as requested in the simulation notes
			fast_fading_sigma_db=1.6,
			deep_fade_prob=0.32,
			deep_fade_penalty_db=4.0,
			ar_coeff=0.70,
			doppler_hz=25.0,
			delay_spread_us=3.0,
		),
		"taxiing": ScenarioConfig(
			name="Taxiing",
			mean_cinr_db=15.5,
			k_factor_db=k_taxiing,
			eta=3.0,
			shadow_sigma_db=1.2,
			fast_fading_sigma_db=1.0,
			deep_fade_prob=0.08,
			deep_fade_penalty_db=2.5,
			ar_coeff=0.55,
			doppler_hz=438.0,  # Adjusted to 438 Hz Doppler for 50 knots at 5 GHz
			delay_spread_us=1.5,
		),
	}


def select_mcs(snr_db: float, mcs_table: list[MCSConfig]) -> MCSConfig:
	eligible = [item for item in mcs_table if snr_db >= item.snr_threshold_db]
	if eligible:
		return eligible[-1]
	return mcs_table[0]


def effective_ebn0_db(snr_db: float, mcs: MCSConfig) -> float:
	gain = mcs.bits_per_symbol * mcs.code_rate * mcs.repetition
	return snr_db + 10.0 * log10(gain)


def qpsk_ber(ebn0_linear: float) -> float:
	return 0.5 * erfc(sqrt(ebn0_linear))


def square_mqam_ber(modulation_order: int, ebn0_linear: float) -> float:
	bits_per_symbol = log2_int(modulation_order)
	if modulation_order == 4:
		return qpsk_ber(ebn0_linear)
	prefactor = (4.0 / bits_per_symbol) * (1.0 - 1.0 / sqrt(modulation_order))
	argument = sqrt((3.0 * bits_per_symbol / (modulation_order - 1.0)) * ebn0_linear)
	return prefactor * 0.5 * erfc(argument / sqrt(2.0))


def ber_for_mcs(snr_db: float, mcs: MCSConfig) -> float:
	ebn0_db = effective_ebn0_db(snr_db, mcs)
	ebn0_linear = 10.0 ** (ebn0_db / 10.0)
	ber = square_mqam_ber(mcs.modulation_order, ebn0_linear)
	if mcs.repetition > 1:
		ber *= 0.72 ** (mcs.repetition - 1)
		ber *= 0.5 ** (mcs.repetition - 1)
	return float(np.clip(ber, 1e-15, 0.5))


def awgn_ber_curve(mcs: MCSConfig, snr_range_db: np.ndarray) -> np.ndarray:
	"""Compute theoretical BER over an AWGN SNR range for a fixed MCS."""
	ber_values = np.empty_like(snr_range_db, dtype=float)
	for index, snr_db in enumerate(snr_range_db):
		ber_values[index] = ber_for_mcs(snr_db, mcs)
	return ber_values


def simulate_awgn_ber(mcs: MCSConfig, snr_db: float, num_bits: int = 10000000, seed: int = 42) -> float:
	"""Simulate BER in AWGN using binomial Monte Carlo simulation."""
	rng = np.random.default_rng(seed)
	p = ber_for_mcs(snr_db, mcs)
	errors = rng.binomial(num_bits, p)
	return float(errors / num_bits)


def simulate_channel_series(
	scenario: ScenarioConfig,
	frames: int,
	rng: np.random.Generator,
) -> np.ndarray:
	snr_series = np.empty(frames, dtype=float)
	base_cinr = scenario.mean_cinr_db + 0.35 * scenario.k_factor_db - 1.2 * (scenario.eta - 3.0)
	prev_delta = 0.0
	doppler_scale = min(1.0, scenario.doppler_hz / 200.0)
	for index in range(frames):
		slow_shadow = rng.normal(0.0, scenario.shadow_sigma_db)
		fast_fading = rng.normal(0.0, scenario.fast_fading_sigma_db * (1.0 + 0.5 * doppler_scale))
		deep_fade = -scenario.deep_fade_penalty_db if rng.random() < scenario.deep_fade_prob else 0.0
		prev_delta = scenario.ar_coeff * prev_delta + (1.0 - scenario.ar_coeff) * (slow_shadow + fast_fading + deep_fade)
		snr_series[index] = np.clip(base_cinr + prev_delta, -5.0, 30.0)
	return snr_series


def estimate_scenario_ber_curve(
	scenario: ScenarioConfig,
	mcs: MCSConfig,
	snr_range_db: np.ndarray,
	samples_per_point: int,
	seed: int,
	esp_loss_db: float = 0.0,
) -> np.ndarray:
	"""Estimate average BER for a fixed MCS over a scenario-specific SNR range (with optional estimation loss)."""
	rng = np.random.default_rng(seed)
	ber_curve = np.empty_like(snr_range_db, dtype=float)
	for index, snr_db in enumerate(snr_range_db):
		scenario_point = ScenarioConfig(
			name=scenario.name,
			mean_cinr_db=snr_db,
			k_factor_db=scenario.k_factor_db,
			eta=scenario.eta,
			shadow_sigma_db=scenario.shadow_sigma_db,
			fast_fading_sigma_db=scenario.fast_fading_sigma_db,
			deep_fade_prob=scenario.deep_fade_prob,
			deep_fade_penalty_db=scenario.deep_fade_penalty_db,
			ar_coeff=scenario.ar_coeff,
			doppler_hz=scenario.doppler_hz,
			delay_spread_us=scenario.delay_spread_us,
		)
		snr_series = simulate_channel_series(scenario_point, samples_per_point, rng)
		# Actual BER is computed by applying the estimation loss at the receiver
		ber_curve[index] = float(np.mean([ber_for_mcs(float(x) - esp_loss_db, mcs) for x in snr_series]))
	return ber_curve


def estimate_amc_ber_curve(
	scenario: ScenarioConfig,
	mcs_table: list[MCSConfig],
	snr_range_db: np.ndarray,
	samples_per_point: int,
	seed: int,
	esp_loss_db: float = 0.0,
) -> np.ndarray:
	"""Estimate average BER for the AMC system (dynamic MCS selection) over SNR range."""
	rng = np.random.default_rng(seed)
	ber_curve = np.empty_like(snr_range_db, dtype=float)
	for index, snr_db in enumerate(snr_range_db):
		scenario_point = ScenarioConfig(
			name=scenario.name,
			mean_cinr_db=snr_db,
			k_factor_db=scenario.k_factor_db,
			eta=scenario.eta,
			shadow_sigma_db=scenario.shadow_sigma_db,
			fast_fading_sigma_db=scenario.fast_fading_sigma_db,
			deep_fade_prob=scenario.deep_fade_prob,
			deep_fade_penalty_db=scenario.deep_fade_penalty_db,
			ar_coeff=scenario.ar_coeff,
			doppler_hz=scenario.doppler_hz,
			delay_spread_us=scenario.delay_spread_us,
		)
		snr_series = simulate_channel_series(scenario_point, samples_per_point, rng)
		ber_list = []
		for x in snr_series:
			# Receiver selects MCS based on estimated SNR (which has esp_loss_db)
			selected = select_mcs(float(x) - esp_loss_db, mcs_table)
			# Actual transmission BER is computed at estimated SNR (reflecting estimation loss)
			ber_val = ber_for_mcs(float(x) - esp_loss_db, selected)
			ber_list.append(ber_val)
		ber_curve[index] = float(np.mean(ber_list))
	return ber_curve


def throughput_mbps(mcs: MCSConfig) -> float:
	spectral_efficiency = (mcs.bits_per_symbol * mcs.code_rate) / mcs.repetition
	max_spectral_efficiency = 6.0 * 0.75
	return PEAK_THROUGHPUT_MBPS * spectral_efficiency / max_spectral_efficiency


def goodput_mbps(mcs: MCSConfig, ber: float, packet_size_bits: int = 4096) -> float:
	ber = np.clip(ber, 1e-15, 1.0)
	per = 1.0 - (1.0 - ber) ** packet_size_bits
	return throughput_mbps(mcs) * (1.0 - per)


def estimate_scenario_throughput_goodput_curves(
	scenario: ScenarioConfig,
	mcs: MCSConfig,
	snr_range_db: np.ndarray,
	samples_per_point: int,
	seed: int,
	packet_size_bits: int = 4096,
	esp_loss_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
	rng = np.random.default_rng(seed)
	throughput_curve = np.full_like(snr_range_db, throughput_mbps(mcs), dtype=float)
	goodput_curve = np.empty_like(snr_range_db, dtype=float)
	for index, snr_db in enumerate(snr_range_db):
		scenario_point = ScenarioConfig(
			name=scenario.name,
			mean_cinr_db=snr_db,
			k_factor_db=scenario.k_factor_db,
			eta=scenario.eta,
			shadow_sigma_db=scenario.shadow_sigma_db,
			fast_fading_sigma_db=scenario.fast_fading_sigma_db,
			deep_fade_prob=scenario.deep_fade_prob,
			deep_fade_penalty_db=scenario.deep_fade_penalty_db,
			ar_coeff=scenario.ar_coeff,
			doppler_hz=scenario.doppler_hz,
			delay_spread_us=scenario.delay_spread_us,
		)
		snr_series = simulate_channel_series(scenario_point, samples_per_point, rng)
		gp_list = []
		for x in snr_series:
			ber_val = ber_for_mcs(float(x) - esp_loss_db, mcs)
			gp_val = goodput_mbps(mcs, ber_val, packet_size_bits)
			gp_list.append(gp_val)
		goodput_curve[index] = float(np.mean(gp_list))
	return throughput_curve, goodput_curve


def estimate_amc_throughput_goodput_curves(
	scenario: ScenarioConfig,
	mcs_table: list[MCSConfig],
	snr_range_db: np.ndarray,
	samples_per_point: int,
	seed: int,
	packet_size_bits: int = 4096,
	esp_loss_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
	rng = np.random.default_rng(seed)
	throughput_curve = np.empty_like(snr_range_db, dtype=float)
	goodput_curve = np.empty_like(snr_range_db, dtype=float)
	for index, snr_db in enumerate(snr_range_db):
		scenario_point = ScenarioConfig(
			name=scenario.name,
			mean_cinr_db=snr_db,
			k_factor_db=scenario.k_factor_db,
			eta=scenario.eta,
			shadow_sigma_db=scenario.shadow_sigma_db,
			fast_fading_sigma_db=scenario.fast_fading_sigma_db,
			deep_fade_prob=scenario.deep_fade_prob,
			deep_fade_penalty_db=scenario.deep_fade_penalty_db,
			ar_coeff=scenario.ar_coeff,
			doppler_hz=scenario.doppler_hz,
			delay_spread_us=scenario.delay_spread_us,
		)
		snr_series = simulate_channel_series(scenario_point, samples_per_point, rng)
		tp_list = []
		gp_list = []
		for x in snr_series:
			selected = select_mcs(float(x) - esp_loss_db, mcs_table)
			ber_val = ber_for_mcs(float(x) - esp_loss_db, selected)
			tp_list.append(throughput_mbps(selected))
			gp_list.append(goodput_mbps(selected, ber_val, packet_size_bits))
		throughput_curve[index] = float(np.mean(tp_list))
		goodput_curve[index] = float(np.mean(gp_list))
	return throughput_curve, goodput_curve


# ---------------------------------------------------------
# SMOOTHING UTILITIES (Savitzky-Golay Filter)
# ---------------------------------------------------------

def savitzky_golay(y: np.ndarray, window_size: int, order: int, deriv: int = 0, rate: int = 1) -> np.ndarray:
	"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter."""
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomial's order")
	half_window = (window_size - 1) // 2
	x = np.arange(-half_window, half_window + 1)
	b = np.vstack([x**i for i in range(order + 1)]).T
	pinv_b = np.linalg.pinv(b)
	from math import factorial
	m = pinv_b[deriv] * (rate**deriv * factorial(deriv))
	firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
	lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
	padded_y = np.concatenate((firstvals, y, lastvals))
	return np.convolve(m[::-1], padded_y, mode='valid')


def smooth_curve(y: np.ndarray, window_size: int = 11, order: int = 2) -> np.ndarray:
	"""Smooth a curve using Savitzky-Golay filter in the log10 domain."""
	log_y = np.log10(y)
	smoothed_log_y = savitzky_golay(log_y, window_size, order)
	return 10.0 ** smoothed_log_y


# ---------------------------------------------------------
# NEW PLOTTING FUNCTIONS
# ---------------------------------------------------------

def plot_fig1_awgn_theory(output_dir: Path, show_plots: bool) -> Path:
	"""Figure 1: Theoretical BER limit in AWGN channel"""
	snr_range_db = np.arange(0.0, 25.5, 0.5)
	mcs_list = [
		MCSConfig("QPSK_1_2_REP4", 4, 0.5, 4, 0.0),
		MCSConfig("QPSK_1_2_REP2", 4, 0.5, 2, 3.0),
		MCSConfig("QPSK_1_2", 4, 0.5, 1, 5.0),
		MCSConfig("16QAM_1_2", 16, 0.5, 1, 10.5),
		MCSConfig("16QAM_3_4", 16, 0.75, 1, 13.4),
		MCSConfig("64QAM_3_4", 64, 0.75, 1, 17.8),
	]
	
	colors = ['magenta', 'cyan', 'blue', 'green', 'orange', 'red']
	
	fig, ax = plt.subplots(figsize=(10, 6))
	for mcs, color in zip(mcs_list, colors):
		ber_curve = awgn_ber_curve(mcs, snr_range_db)
		ax.semilogy(snr_range_db, ber_curve, label=mcs.name, color=color, linewidth=2.3)
		
	ax.axhline(TARGET_BER, color="black", linestyle="--", linewidth=1.2, label=f"Target BER = {TARGET_BER:.0e}")
	ax.set_title("Figure 1: Theoretical BER Limit of MCS Schemes in AWGN Channel")
	ax.set_xlabel("Signal-to-Noise Ratio SNR / CINR (dB)")
	ax.set_ylabel("Bit Error Rate (BER)")
	ax.set_ylim(1e-12, 1e-1)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	ax.legend(loc="upper right", fontsize=9)
	
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig1_awgn_theory.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def plot_fig2_awgn_simulation(output_dir: Path, show_plots: bool) -> Path:
	"""Figure 2: BER performance in AWGN channel (theory vs. simulation)"""
	snr_range_db = np.arange(0.0, 25.5, 0.5)
	snr_sim_points = np.arange(0.0, 25.5, 2.0)
	
	mcs_selections = [
		(MCSConfig("QPSK_1_2", 4, 0.5, 1, 5.0), 'blue', 'o', 'QPSK 1/2'),
		(MCSConfig("16QAM_1_2", 16, 0.5, 1, 10.5), 'green', 's', '16-QAM 1/2'),
		(MCSConfig("64QAM_3_4", 64, 0.75, 1, 17.8), 'red', '^', '64-QAM 3/4')
	]
	
	fig, ax = plt.subplots(figsize=(10, 6))
	for mcs, color, marker, label_name in mcs_selections:
		# Plot theoretical lines
		ber_theory = awgn_ber_curve(mcs, snr_range_db)
		ax.semilogy(snr_range_db, ber_theory, color=color, linestyle='-', linewidth=2.3, label=f"Theory {label_name}")
		
		# Compute and plot simulated points
		sim_points = []
		for snr in snr_sim_points:
			sim_val = simulate_awgn_ber(mcs, snr, num_bits=5000000, seed=42)
			sim_points.append(sim_val)
		# Filter out zero values for correct log plot representation
		sim_points = np.array(sim_points)
		valid_idx = sim_points > 0
		ax.scatter(snr_sim_points[valid_idx], sim_points[valid_idx], color=color, marker=marker, s=50, label=f"Simulation {label_name}", zorder=3)
		
	ax.axhline(TARGET_BER, color="black", linestyle="--", linewidth=1.2, label=f"Target BER = {TARGET_BER:.0e}")
	ax.set_title("Figure 2: BER Performance in AWGN Channel (Theory vs. Monte Carlo Simulation)")
	ax.set_xlabel("Signal-to-Noise Ratio SNR / CINR (dB)")
	ax.set_ylabel("Bit Error Rate (BER)")
	ax.set_ylim(1e-7, 1e-1)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	ax.legend(loc="upper right", fontsize=9)
	
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig2_awgn_theory_vs_sim.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def plot_fig3_parking_ber(scenario: ScenarioConfig, output_dir: Path, show_plots: bool) -> Path:
	"""Figure 3: BER performance in Parking scenario (NLOS)"""
	snr_range_db = np.arange(-5.0, 20.5, 0.5)
	mcs_list = [
		MCSConfig("QPSK_1_2_REP4", 4, 0.5, 4, 0.0),
		MCSConfig("QPSK_1_2_REP2", 4, 0.5, 2, 3.0),
		MCSConfig("QPSK_1_2", 4, 0.5, 1, 5.0),
		MCSConfig("16QAM_1_2", 16, 0.5, 1, 10.5),
		MCSConfig("16QAM_3_4", 16, 0.75, 1, 13.4),
	]
	
	colors = ['magenta', 'cyan', 'blue', 'green', 'orange']
	
	fig, ax = plt.subplots(figsize=(11, 7))
	
	for mcs, color in zip(mcs_list, colors):
		# 1. AWGN Theory (dotted)
		ber_awgn = awgn_ber_curve(mcs, snr_range_db)
		ax.semilogy(snr_range_db, ber_awgn, color=color, linestyle=':', linewidth=1.2)
		
		# 2. Fading Simulation - PCK (solid)
		ber_pck = estimate_scenario_ber_curve(scenario, mcs, snr_range_db, samples_per_point=SAMPLES_PER_POINT, seed=42, esp_loss_db=0.0)
		ber_pck_smooth = smooth_curve(ber_pck)
		ax.semilogy(snr_range_db, ber_pck_smooth, color=color, linestyle='-', linewidth=2.0)
		
		# 3. Fading Simulation - ESP (dashed, 2 dB loss)
		ber_esp = estimate_scenario_ber_curve(scenario, mcs, snr_range_db, samples_per_point=SAMPLES_PER_POINT, seed=42, esp_loss_db=2.0)
		ber_esp_smooth = smooth_curve(ber_esp)
		ax.semilogy(snr_range_db, ber_esp_smooth, color=color, linestyle='--', linewidth=1.6)
		
	# Typical Parking SNR range shading
	ax.axvspan(2.0, 6.0, color="gray", alpha=0.12, label="Typical parking operating region (2-6 dB)")
	
	# Horizontal target lines
	ax.axhline(TARGET_BER, color="black", linestyle="--", linewidth=1.6, label=f"Target BER = {TARGET_BER:.0e}")
	ax.axhline(1e-3, color="red", linestyle="-.", linewidth=1.6, label="Target parking BER threshold = 10^-3")
	
	# Design custom legend to keep it clean and organized
	color_handles = []
	for mcs, color in zip(mcs_list, colors):
		color_handles.append(mlines.Line2D([], [], color=color, label=mcs.name, linewidth=2.0))
		
	style_handles = [
		mlines.Line2D([], [], color='black', linestyle='-', label='Fading Simulation (PCK) (Solid line)', linewidth=1.5),
		mlines.Line2D([], [], color='black', linestyle='--', label='Fading Simulation (ESP, -2 dB) (Dashed line)', linewidth=1.5),
		mlines.Line2D([], [], color='black', linestyle=':', label='AWGN Channel (Theory) (Dotted line)', linewidth=1.2)
	]
	
	ax.set_title(f"Figure 3: BER Performance in Parking Scenario (NLOS, K = {scenario.k_factor_db:.1f} dB, σ = {scenario.shadow_sigma_db:.1f} dB)")
	ax.set_xlabel("Signal-to-Noise Ratio SNR / CINR (dB)")
	ax.set_ylabel("Bit Error Rate (BER)")
	ax.set_ylim(1e-6, 1e-1)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	
	# Place the legend
	ax.legend(handles=color_handles + style_handles, loc="upper right", fontsize=8)
	
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig3_parking_ber_vs_snr.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def plot_fig4_taxiing_ber(scenario: ScenarioConfig, output_dir: Path, show_plots: bool) -> Path:
	"""Figure 4: BER performance in Taxiing scenario (LOS)"""
	snr_range_db = np.arange(5.0, 25.5, 0.5)
	mcs_list = [
		MCSConfig("16QAM_1_2", 16, 0.5, 1, 10.5),
		MCSConfig("16QAM_3_4", 16, 0.75, 1, 13.4),
		MCSConfig("64QAM_3_4", 64, 0.75, 1, 17.8),
	]
	
	colors = ['green', 'orange', 'red']
	
	fig, ax = plt.subplots(figsize=(11, 7))
	
	for mcs, color in zip(mcs_list, colors):
		# 1. AWGN Theory (dotted)
		ber_awgn = awgn_ber_curve(mcs, snr_range_db)
		ax.semilogy(snr_range_db, ber_awgn, color=color, linestyle=':', linewidth=1.2)
		
		# 2. Fading Simulation - PCK (solid)
		ber_pck = estimate_scenario_ber_curve(scenario, mcs, snr_range_db, samples_per_point=SAMPLES_PER_POINT, seed=42, esp_loss_db=0.0)
		ber_pck_smooth = smooth_curve(ber_pck)
		ax.semilogy(snr_range_db, ber_pck_smooth, color=color, linestyle='-', linewidth=2.0)
		
		# 3. Fading Simulation - ESP (dashed, 2 dB loss)
		ber_esp = estimate_scenario_ber_curve(scenario, mcs, snr_range_db, samples_per_point=SAMPLES_PER_POINT, seed=42, esp_loss_db=2.0)
		ber_esp_smooth = smooth_curve(ber_esp)
		ax.semilogy(snr_range_db, ber_esp_smooth, color=color, linestyle='--', linewidth=1.6)
		
	# Typical Taxiing SNR range shading
	ax.axvspan(15.0, 18.0, color="gray", alpha=0.12, label="Typical taxiing operating region (15-18 dB)")
	
	# Horizontal target line
	ax.axhline(TARGET_BER, color="black", linestyle="--", linewidth=1.2, label=f"Target BER = {TARGET_BER:.0e}")
	
	# Design custom legend to keep it clean and organized
	color_handles = []
	for mcs, color in zip(mcs_list, colors):
		color_handles.append(mlines.Line2D([], [], color=color, label=mcs.name, linewidth=2.0))
		
	style_handles = [
		mlines.Line2D([], [], color='black', linestyle='-', label='Fading Simulation (PCK) (Solid line)', linewidth=1.5),
		mlines.Line2D([], [], color='black', linestyle='--', label='Fading Simulation (ESP, -2 dB) (Dashed line)', linewidth=1.5),
		mlines.Line2D([], [], color='black', linestyle=':', label='AWGN Channel (Theory) (Dotted line)', linewidth=1.2)
	]
	
	ax.set_title(f"Figure 4: BER Performance in Taxiing Scenario (LOS, K = {scenario.k_factor_db:.1f} dB, Doppler = {scenario.doppler_hz:.1f} Hz)")
	ax.set_xlabel("Signal-to-Noise Ratio SNR / CINR (dB)")
	ax.set_ylabel("Bit Error Rate (BER)")
	ax.set_ylim(1e-12, 1e-1)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	
	# Place the legend
	ax.legend(handles=color_handles + style_handles, loc="upper right", fontsize=8)
	
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig4_taxiing_ber_vs_snr.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def plot_fig5_amc_comparison(scenarios: dict[str, ScenarioConfig], mcs_table: list[MCSConfig], output_dir: Path, show_plots: bool) -> Path:
	"""Figure 5: Comparison of BER performance of AMC and optimal MCS schemes"""
	snr_range_db = np.arange(-5.0, 25.5, 0.5)
	
	parking_scenario = scenarios["parking"]
	taxiing_scenario = scenarios["taxiing"]
	
	# Get baseline MCS for Parking (QPSK 1/2 REP2) and Taxiing (64QAM 3/4)
	qpsk_rep2_mcs = next(mcs for mcs in mcs_table if mcs.name == "QPSK_1_2_REP2")
	qam64_mcs = next(mcs for mcs in mcs_table if mcs.name == "64QAM_3_4")
	
	fig, ax = plt.subplots(figsize=(12, 8))
	
	# 1. Parking optimal fixed MCS (QPSK 1/2 REP2)
	parking_fixed_pck = estimate_scenario_ber_curve(parking_scenario, qpsk_rep2_mcs, snr_range_db, samples_per_point=SAMPLES_PER_POINT, seed=42, esp_loss_db=0.0)
	parking_fixed_esp = estimate_scenario_ber_curve(parking_scenario, qpsk_rep2_mcs, snr_range_db, samples_per_point=SAMPLES_PER_POINT, seed=42, esp_loss_db=2.0)
	ax.semilogy(snr_range_db, smooth_curve(parking_fixed_pck), 'red', linestyle='-', alpha=0.5, linewidth=1.8, label='Parking: QPSK 1/2 REP2 (PCK)')
	ax.semilogy(snr_range_db, smooth_curve(parking_fixed_esp), 'red', linestyle='--', alpha=0.5, linewidth=1.6, label='Parking: QPSK 1/2 REP2 (ESP)')
	
	# 2. Taxiing optimal fixed MCS (64-QAM 3/4)
	taxiing_fixed_pck = estimate_scenario_ber_curve(taxiing_scenario, qam64_mcs, snr_range_db, samples_per_point=SAMPLES_PER_POINT, seed=42, esp_loss_db=0.0)
	taxiing_fixed_esp = estimate_scenario_ber_curve(taxiing_scenario, qam64_mcs, snr_range_db, samples_per_point=SAMPLES_PER_POINT, seed=42, esp_loss_db=2.0)
	ax.semilogy(snr_range_db, smooth_curve(taxiing_fixed_pck), 'blue', linestyle='-', alpha=0.5, linewidth=1.8, label='Taxiing: 64-QAM 3/4 (PCK)')
	ax.semilogy(snr_range_db, smooth_curve(taxiing_fixed_esp), 'blue', linestyle='--', alpha=0.5, linewidth=1.6, label='Taxiing: 64-QAM 3/4 (ESP)')
	
	# 3. Dynamic AMC BER curves
	amc_parking_pck = estimate_amc_ber_curve(parking_scenario, mcs_table, snr_range_db, samples_per_point=SAMPLES_PER_POINT, seed=42, esp_loss_db=0.0)
	amc_parking_esp = estimate_amc_ber_curve(parking_scenario, mcs_table, snr_range_db, samples_per_point=SAMPLES_PER_POINT, seed=42, esp_loss_db=2.0)
	ax.semilogy(snr_range_db, smooth_curve(amc_parking_pck), 'green', linestyle='-', linewidth=2.8, label='Parking: AMC System (PCK)')
	ax.semilogy(snr_range_db, smooth_curve(amc_parking_esp), 'green', linestyle='--', linewidth=2.3, label='Parking: AMC System (ESP)')
	
	amc_taxiing_pck = estimate_amc_ber_curve(taxiing_scenario, mcs_table, snr_range_db, samples_per_point=SAMPLES_PER_POINT, seed=42, esp_loss_db=0.0)
	amc_taxiing_esp = estimate_amc_ber_curve(taxiing_scenario, mcs_table, snr_range_db, samples_per_point=SAMPLES_PER_POINT, seed=42, esp_loss_db=2.0)
	ax.semilogy(snr_range_db, smooth_curve(amc_taxiing_pck), 'darkorange', linestyle='-', linewidth=2.8, label='Taxiing: AMC System (PCK)')
	ax.semilogy(snr_range_db, smooth_curve(amc_taxiing_esp), 'darkorange', linestyle='--', linewidth=2.3, label='Taxiing: AMC System (ESP)')
	
	# Horizontal target line
	ax.axhline(TARGET_BER, color="black", linestyle="--", linewidth=1.6, label=f"Target BER = {TARGET_BER:.0e}")
	
	# Vertical lines indicating AMC switching thresholds
	thresholds = [mcs.snr_threshold_db for mcs in mcs_table if mcs.snr_threshold_db > 0.0]
	thresholds = sorted(list(set(thresholds)))
	for t in thresholds:
		ax.axvline(t, color='gray', linestyle=':', alpha=0.6)
		ax.text(t + 0.2, 5e-12, f"{t} dB", rotation=90, verticalalignment='bottom', fontsize=8, color='dimgray')
		
	ax.set_title("Figure 5: Comparison of BER Performance of Adaptive AMC System and Fixed MCS Configurations")
	ax.set_xlabel("Signal-to-Noise Ratio SNR / CINR (dB)")
	ax.set_ylabel("Bit Error Rate (BER)")
	ax.set_ylim(1e-12, 1e-1)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	ax.legend(loc="upper right", fontsize=8.5)
	
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig5_amc_comparison.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def plot_fig6_throughput_vs_snr(
	scenarios: dict[str, ScenarioConfig],
	mcs_table: list[MCSConfig],
	output_dir: Path,
	show_plots: bool,
) -> Path:
	"""Figure 6: Comparison of throughput between AMC and fixed MCS schemes"""
	snr_range_db = np.arange(-5.0, 25.5, 0.5)
	parking_scenario = scenarios["parking"]
	taxiing_scenario = scenarios["taxiing"]
	
	qpsk_rep2_mcs = next(mcs for mcs in mcs_table if mcs.name == "QPSK_1_2_REP2")
	qam64_mcs = next(mcs for mcs in mcs_table if mcs.name == "64QAM_3_4")
	
	fig, ax = plt.subplots(figsize=(12, 8))
	
	# 1. Parking Fixed (QPSK 1/2 REP2)
	p_fixed_tp, _ = estimate_scenario_throughput_goodput_curves(parking_scenario, qpsk_rep2_mcs, snr_range_db, SAMPLES_PER_POINT, 42, esp_loss_db=0.0)
	p_fixed_tp_esp, _ = estimate_scenario_throughput_goodput_curves(parking_scenario, qpsk_rep2_mcs, snr_range_db, SAMPLES_PER_POINT, 42, esp_loss_db=2.0)
	ax.plot(snr_range_db, p_fixed_tp, 'red', linestyle='-', alpha=0.5, linewidth=1.8, label='Parking: QPSK 1/2 REP2 (PCK)')
	ax.plot(snr_range_db, p_fixed_tp_esp, 'red', linestyle='--', alpha=0.5, linewidth=1.6, label='Parking: QPSK 1/2 REP2 (ESP)')
	
	# 2. Taxiing Fixed (64-QAM 3/4)
	t_fixed_tp, _ = estimate_scenario_throughput_goodput_curves(taxiing_scenario, qam64_mcs, snr_range_db, SAMPLES_PER_POINT, 42, esp_loss_db=0.0)
	t_fixed_tp_esp, _ = estimate_scenario_throughput_goodput_curves(taxiing_scenario, qam64_mcs, snr_range_db, SAMPLES_PER_POINT, 42, esp_loss_db=2.0)
	ax.plot(snr_range_db, t_fixed_tp, 'blue', linestyle='-', alpha=0.5, linewidth=1.8, label='Taxiing: 64-QAM 3/4 (PCK)')
	ax.plot(snr_range_db, t_fixed_tp_esp, 'blue', linestyle='--', alpha=0.5, linewidth=1.6, label='Taxiing: 64-QAM 3/4 (ESP)')
	
	# 3. AMC Parking
	amc_p_tp, _ = estimate_amc_throughput_goodput_curves(parking_scenario, mcs_table, snr_range_db, SAMPLES_PER_POINT, 42, esp_loss_db=0.0)
	amc_p_tp_esp, _ = estimate_amc_throughput_goodput_curves(parking_scenario, mcs_table, snr_range_db, SAMPLES_PER_POINT, 42, esp_loss_db=2.0)
	ax.plot(snr_range_db, amc_p_tp, 'green', linestyle='-', linewidth=2.8, label='Parking: AMC System (PCK)')
	ax.plot(snr_range_db, amc_p_tp_esp, 'green', linestyle='--', linewidth=2.3, label='Parking: AMC System (ESP)')
	
	# 4. AMC Taxiing
	amc_t_tp, _ = estimate_amc_throughput_goodput_curves(taxiing_scenario, mcs_table, snr_range_db, SAMPLES_PER_POINT, 42, esp_loss_db=0.0)
	amc_t_tp_esp, _ = estimate_amc_throughput_goodput_curves(taxiing_scenario, mcs_table, snr_range_db, SAMPLES_PER_POINT, 42, esp_loss_db=2.0)
	ax.plot(snr_range_db, amc_t_tp, 'darkorange', linestyle='-', linewidth=2.8, label='Taxiing: AMC System (PCK)')
	ax.plot(snr_range_db, amc_t_tp_esp, 'darkorange', linestyle='--', linewidth=2.3, label='Taxiing: AMC System (ESP)')
	
	ax.set_title("Figure 6: Comparison of Throughput of Adaptive AMC System and Fixed MCS Configurations")
	ax.set_xlabel("Signal-to-Noise Ratio SNR / CINR (dB)")
	ax.set_ylabel("Theoretical throughput (Mbps)")
	ax.set_ylim(-0.2, 8.0)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	ax.legend(loc="lower right", fontsize=8.5)
	
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig6_amc_throughput_comparison.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def plot_fig7_goodput_vs_snr(
	scenarios: dict[str, ScenarioConfig],
	mcs_table: list[MCSConfig],
	output_dir: Path,
	show_plots: bool,
	packet_size_bits: int = 4096,
) -> Path:
	"""Figure 7: Comparison of actual performance (Goodput) between AMC and fixed MCS schemes"""
	snr_range_db = np.arange(-5.0, 25.5, 0.5)
	parking_scenario = scenarios["parking"]
	taxiing_scenario = scenarios["taxiing"]
	
	qpsk_rep2_mcs = next(mcs for mcs in mcs_table if mcs.name == "QPSK_1_2_REP2")
	qam64_mcs = next(mcs for mcs in mcs_table if mcs.name == "64QAM_3_4")
	
	fig, ax = plt.subplots(figsize=(12, 8))
	
	# 1. Parking Fixed (QPSK 1/2 REP2)
	_, p_fixed_gp = estimate_scenario_throughput_goodput_curves(parking_scenario, qpsk_rep2_mcs, snr_range_db, SAMPLES_PER_POINT, 42, packet_size_bits, esp_loss_db=0.0)
	_, p_fixed_gp_esp = estimate_scenario_throughput_goodput_curves(parking_scenario, qpsk_rep2_mcs, snr_range_db, SAMPLES_PER_POINT, 42, packet_size_bits, esp_loss_db=2.0)
	ax.plot(snr_range_db, p_fixed_gp, 'red', linestyle='-', alpha=0.5, linewidth=1.8, label='Parking: QPSK 1/2 REP2 (PCK)')
	ax.plot(snr_range_db, p_fixed_gp_esp, 'red', linestyle='--', alpha=0.5, linewidth=1.6, label='Parking: QPSK 1/2 REP2 (ESP)')
	
	# 2. Taxiing Fixed (64-QAM 3/4)
	_, t_fixed_gp = estimate_scenario_throughput_goodput_curves(taxiing_scenario, qam64_mcs, snr_range_db, SAMPLES_PER_POINT, 42, packet_size_bits, esp_loss_db=0.0)
	_, t_fixed_gp_esp = estimate_scenario_throughput_goodput_curves(taxiing_scenario, qam64_mcs, snr_range_db, SAMPLES_PER_POINT, 42, packet_size_bits, esp_loss_db=2.0)
	ax.plot(snr_range_db, t_fixed_gp, 'blue', linestyle='-', alpha=0.5, linewidth=1.8, label='Taxiing: 64-QAM 3/4 (PCK)')
	ax.plot(snr_range_db, t_fixed_gp_esp, 'blue', linestyle='--', alpha=0.5, linewidth=1.6, label='Taxiing: 64-QAM 3/4 (ESP)')
	
	# 3. AMC Parking
	_, amc_p_gp = estimate_amc_throughput_goodput_curves(parking_scenario, mcs_table, snr_range_db, SAMPLES_PER_POINT, 42, packet_size_bits, esp_loss_db=0.0)
	_, amc_p_gp_esp = estimate_amc_throughput_goodput_curves(parking_scenario, mcs_table, snr_range_db, SAMPLES_PER_POINT, 42, packet_size_bits, esp_loss_db=2.0)
	ax.plot(snr_range_db, amc_p_gp, 'green', linestyle='-', linewidth=2.8, label='Parking: AMC System (PCK)')
	ax.plot(snr_range_db, amc_p_gp_esp, 'green', linestyle='--', linewidth=2.3, label='Parking: AMC System (ESP)')
	
	# 4. AMC Taxiing
	_, amc_t_gp = estimate_amc_throughput_goodput_curves(taxiing_scenario, mcs_table, snr_range_db, SAMPLES_PER_POINT, 42, packet_size_bits, esp_loss_db=0.0)
	_, amc_t_gp_esp = estimate_amc_throughput_goodput_curves(taxiing_scenario, mcs_table, snr_range_db, SAMPLES_PER_POINT, 42, packet_size_bits, esp_loss_db=2.0)
	ax.plot(snr_range_db, amc_t_gp, 'darkorange', linestyle='-', linewidth=2.8, label='Taxiing: AMC System (PCK)')
	ax.plot(snr_range_db, amc_t_gp_esp, 'darkorange', linestyle='--', linewidth=2.3, label='Taxiing: AMC System (ESP)')
	
	ax.set_title(f"Figure 7: Comparison of Actual Performance (Goodput) of AMC and Fixed Configurations (Packet = {packet_size_bits} bits)")
	ax.set_xlabel("Signal-to-Noise Ratio SNR / CINR (dB)")
	ax.set_ylabel("Actual throughput Goodput (Mbps)")
	ax.set_ylim(-0.2, 8.0)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	ax.legend(loc="lower right", fontsize=8.5)
	
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig7_amc_goodput_comparison.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def plot_fig8_timeseries_sim(
	scenarios: dict[str, ScenarioConfig],
	mcs_table: list[MCSConfig],
	output_dir: Path,
	show_plots: bool,
	frames: int = 200,
	packet_size_bits: int = 4096,
) -> Path:
	"""Figure 8: Time-series simulation of an aircraft at Tan Son Nhat Airport"""
	rng = np.random.default_rng(42)
	
	# Simulate Parking Scenario (NLOS, mean SNR = 8 dB)
	parking_scenario = scenarios["parking"]
	parking_snr = simulate_channel_series(parking_scenario, frames, rng)
	
	# Simulate Taxiing Scenario (LOS, mean SNR = 15.5 dB)
	taxiing_scenario = scenarios["taxiing"]
	taxiing_snr = simulate_channel_series(taxiing_scenario, frames, rng)
	
	# Parking stats
	p_mcs_list = []
	p_ber_theory = []
	p_per_theory = []
	p_throughput = []
	p_goodput = []
	
	# Taxiing stats
	t_mcs_list = []
	t_ber_theory = []
	t_per_theory = []
	t_throughput = []
	t_goodput = []
	
	for x in parking_snr:
		selected = select_mcs(float(x), mcs_table)
		p_mcs_list.append(selected)
		ber = ber_for_mcs(float(x), selected)
		p_ber_theory.append(ber)
		per = 1.0 - (1.0 - ber) ** packet_size_bits
		p_per_theory.append(per)
		tp = throughput_mbps(selected)
		p_throughput.append(tp)
		p_goodput.append(tp * (1.0 - per))
		
	for x in taxiing_snr:
		selected = select_mcs(float(x), mcs_table)
		t_mcs_list.append(selected)
		ber = ber_for_mcs(float(x), selected)
		t_ber_theory.append(ber)
		per = 1.0 - (1.0 - ber) ** packet_size_bits
		t_per_theory.append(per)
		tp = throughput_mbps(selected)
		t_throughput.append(tp)
		t_goodput.append(tp * (1.0 - per))
		
	# Create subplots
	fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
	frame_indices = np.arange(1, frames + 1)
	
	mcs_names = [mcs.name for mcs in mcs_table]
	mcs_order = {name: index for index, name in enumerate(mcs_names)}
	p_mcs_idx = [mcs_order[mcs.name] for mcs in p_mcs_list]
	t_mcs_idx = [mcs_order[mcs.name] for mcs in t_mcs_list]
	
	# Row 0: SNR
	axes[0, 0].plot(frame_indices, parking_snr, color="tab:blue", linewidth=1.5, label="SNR/CINR")
	axes[0, 0].axhline(parking_scenario.mean_cinr_db, color="darkblue", linestyle="--", linewidth=1.0, label="Mean")
	axes[0, 0].set_ylabel("SNR / CINR (dB)")
	axes[0, 0].set_title(f"Parking Scenario (NLOS, Mean = 8.0 dB, K = {parking_scenario.k_factor_db:.1f} dB)")
	axes[0, 0].grid(True, alpha=0.3)
	axes[0, 0].legend(loc="upper right", fontsize=8)
	
	axes[0, 1].plot(frame_indices, taxiing_snr, color="tab:blue", linewidth=1.5, label="SNR/CINR")
	axes[0, 1].axhline(taxiing_scenario.mean_cinr_db, color="darkblue", linestyle="--", linewidth=1.0, label="Mean")
	axes[0, 1].set_ylabel("SNR / CINR (dB)")
	axes[0, 1].set_title(f"Taxiing Scenario (LOS, Mean = 15.5 dB, K = {taxiing_scenario.k_factor_db:.1f} dB)")
	axes[0, 1].grid(True, alpha=0.3)
	axes[0, 1].legend(loc="upper right", fontsize=8)
	
	# Row 1: MCS Selection
	axes[1, 0].step(frame_indices, p_mcs_idx, where="mid", color="tab:orange", linewidth=1.5)
	axes[1, 0].set_yticks(range(len(mcs_names)))
	axes[1, 0].set_yticklabels(mcs_names, fontsize=8)
	axes[1, 0].set_ylabel("Selected MCS")
	axes[1, 0].grid(True, alpha=0.3)
	
	axes[1, 1].step(frame_indices, t_mcs_idx, where="mid", color="tab:orange", linewidth=1.5)
	axes[1, 1].set_yticks(range(len(mcs_names)))
	axes[1, 1].set_yticklabels(mcs_names, fontsize=8)
	axes[1, 1].set_ylabel("Selected MCS")
	axes[1, 1].grid(True, alpha=0.3)
	
	# Row 2: BER & PER
	axes[2, 0].semilogy(frame_indices, p_ber_theory, color="tab:green", linewidth=1.2, label="Theoretical BER")
	axes[2, 0].semilogy(frame_indices, p_per_theory, color="tab:red", linewidth=1.2, label="PER (4096 bits)")
	axes[2, 0].axhline(TARGET_BER, color="black", linestyle="--", linewidth=0.9, label="Target BER = 1e-6")
	axes[2, 0].set_ylabel("Error rate")
	axes[2, 0].grid(True, which="both", alpha=0.3)
	axes[2, 0].legend(loc="upper right", fontsize=8)
	
	axes[2, 1].semilogy(frame_indices, t_ber_theory, color="tab:green", linewidth=1.2, label="Theoretical BER")
	axes[2, 1].semilogy(frame_indices, t_per_theory, color="tab:red", linewidth=1.2, label="PER (4096 bits)")
	axes[2, 1].axhline(TARGET_BER, color="black", linestyle="--", linewidth=0.9, label="Target BER = 1e-6")
	axes[2, 1].set_ylabel("Error rate")
	axes[2, 1].grid(True, which="both", alpha=0.3)
	axes[2, 1].legend(loc="upper right", fontsize=8)
	
	# Row 3: Throughput & Goodput
	axes[3, 0].plot(frame_indices, p_throughput, color="tab:purple", linewidth=1.5, label="Throughput (Raw)")
	axes[3, 0].plot(frame_indices, p_goodput, color="tab:olive", linewidth=1.5, label="Goodput (Effective)")
	axes[3, 0].set_ylabel("Data rate (Mbps)")
	axes[3, 0].set_xlabel("Frame index")
	axes[3, 0].grid(True, alpha=0.3)
	axes[3, 0].legend(loc="upper right", fontsize=8)
	
	axes[3, 1].plot(frame_indices, t_throughput, color="tab:purple", linewidth=1.5, label="Throughput (Raw)")
	axes[3, 1].plot(frame_indices, t_goodput, color="tab:olive", linewidth=1.5, label="Goodput (Effective)")
	axes[3, 1].set_ylabel("Data rate (Mbps)")
	axes[3, 1].set_xlabel("Frame index")
	axes[3, 1].grid(True, alpha=0.3)
	axes[3, 1].legend(loc="upper right", fontsize=8)
	
	plt.suptitle("Figure 8: Real-time adaptive AMC link simulation at Tan Son Nhat Airport", fontsize=14, fontweight="bold")
	fig.tight_layout()
	
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig8_amc_timeseries_simulation.png"
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def fcc_mask_m(freq_offsets_mhz: np.ndarray, bandwidth_mhz: float = 5.0) -> np.ndarray:
	mask_dbc = np.zeros_like(freq_offsets_mhz)
	for idx, fd in enumerate(freq_offsets_mhz):
		p = (abs(fd) / bandwidth_mhz) * 100.0
		if p <= 45.0:
			att = 0.0
		elif p <= 50.0:
			att = 568.0 * log10(p / 45.0)
		elif p <= 55.0:
			att = 26.0 + 145.0 * log10(p / 50.0)
		elif p <= 100.0:
			att = 32.0 + 31.0 * log10(p / 55.0)
		elif p <= 150.0:
			att = 40.0 + 57.0 * log10(p / 100.0)
		else:
			att = 50.0
		mask_dbc[idx] = -att
	return mask_dbc


def plot_fig9_psd_emission_mask(output_dir: Path, show_plots: bool) -> Path:
	"""Figure 9: Emitted power spectral density (PSD) and Mask M spectrum mask (FCC CFR 47 Part 90.210)"""
	fs = 20e6
	nfft = 2048
	nactive = 384
	np.random.seed(42)
	
	num_symbols = 80
	active_indices = np.concatenate([np.arange(-nactive//2, 0), np.arange(1, nactive//2 + 1)])
	
	time_signal = np.array([], dtype=complex)
	for _ in range(num_symbols):
		symbols = (np.random.choice([-1, 1], nactive) + 1j * np.random.choice([-1, 1], nactive)) / sqrt(2)
		freq_data = np.zeros(nfft, dtype=complex)
		freq_data[active_indices % nfft] = symbols
		symbol_time = np.fft.ifft(freq_data) * sqrt(nfft)
		
		cp_len = 256
		symbol_time_cp = np.concatenate([symbol_time[-cp_len:], symbol_time])
		time_signal = np.concatenate([time_signal, symbol_time_cp])
		
	nfft_welch = 1024
	hop = nfft_welch // 2
	window = np.hamming(nfft_welch)
	num_blocks = (len(time_signal) - nfft_welch) // hop + 1
	psd_accum = np.zeros(nfft_welch)
	for i in range(num_blocks):
		block = time_signal[i*hop : i*hop + nfft_welch] * window
		psd_accum += np.abs(np.fft.fft(block))**2
	psd_avg_unfiltered = psd_accum / num_blocks
	psd_avg_unfiltered = np.fft.fftshift(psd_avg_unfiltered)
	psd_db_unfiltered = 10.0 * np.log10(psd_avg_unfiltered + 1e-15)
	
	cutoff_hz = 2.4e6
	num_taps = 127
	t = np.arange(num_taps) - (num_taps - 1) / 2
	h = np.sinc(2 * cutoff_hz * t / fs)
	h *= np.hamming(num_taps)
	h /= np.sum(h)
	time_signal_filtered = np.convolve(time_signal, h, mode='same')
	
	psd_accum_filtered = np.zeros(nfft_welch)
	for i in range(num_blocks):
		block = time_signal_filtered[i*hop : i*hop + nfft_welch] * window
		psd_accum_filtered += np.abs(np.fft.fft(block))**2
	psd_avg_filtered = psd_accum_filtered / num_blocks
	psd_avg_filtered = np.fft.fftshift(psd_avg_filtered)
	psd_db_filtered = 10.0 * np.log10(psd_avg_filtered + 1e-15)
	
	center_idx = nfft_welch // 2
	peak_idx_range = slice(center_idx - 50, center_idx + 50)
	peak_unfiltered = np.max(psd_db_unfiltered[peak_idx_range])
	peak_filtered = np.max(psd_db_filtered[peak_idx_range])
	
	psd_dbc_unfiltered = psd_db_unfiltered - peak_unfiltered
	psd_dbc_filtered = psd_db_filtered - peak_filtered
	
	freq_axis_mhz = np.linspace(-fs/2e6, fs/2e6, nfft_welch)
	mask = fcc_mask_m(freq_axis_mhz, bandwidth_mhz=5.0)
	
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.plot(freq_axis_mhz, psd_dbc_unfiltered, color='tab:red', linestyle=':', alpha=0.75, linewidth=1.8, label='Unfiltered PSD (Non-compliant)')
	ax.plot(freq_axis_mhz, psd_dbc_filtered, color='tab:blue', linestyle='-', linewidth=2.3, label='Filtered PSD (Compliant)')
	ax.plot(freq_axis_mhz, mask, color='black', linestyle='--', linewidth=2.5, label='Emission mask Mask M (FCC § 90.210)')
	
	ax.set_title("Figure 9: AeroMACS emitted power spectral density (PSD) and FCC Mask M at TSN")
	ax.set_xlabel("Frequency offset from center frequency (MHz)")
	ax.set_ylabel("Relative power spectral density (dBc)")
	ax.set_ylim(-70, 10)
	ax.set_xlim(-10, 10)
	ax.grid(True, which="both", alpha=0.35)
	ax.legend(loc="upper right", fontsize=9.5)
	
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig9_psd_mask.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def plot_fig10_mss_protection(output_dir: Path, show_plots: bool) -> Path:
	"""Figure 10: Globalstar MSS mobile satellite protection criteria and interference temperature"""
	re = 6378.0
	h_sat = 1414.0
	f_hz = 5095e6
	c = 3.0e8
	k_boltz = 1.380649e-23
	t_sys = 600.0
	b_sat = 1.23e6
	l_atm = 0.5
	g_sat = 3.0
	
	target_p_rx_dbw = -157.3
	
	theta_deg = np.linspace(5.0, 90.0, 100)
	theta_rad = np.radians(theta_deg)
	
	d_km = re * (np.sqrt(((re + h_sat)/re)**2 - np.cos(theta_rad)**2) - np.sin(theta_rad))
	fspl_db = 20.0 * np.log10(4.0 * np.pi * d_km * 1000.0 * f_hz / c)
	
	theta_tilt = -2.0
	theta_3db = 10.0
	g_max_bs = 15.0
	g_bs = g_max_bs - 12.0 * ((theta_deg - theta_tilt) / theta_3db)**2
	g_bs = np.maximum(g_bs, -15.0)
	
	bw_loss_db = 10.0 * np.log10(5.0 / 1.23)
	
	eirp_bs_dbm_sky = 18.31 + g_bs
	eirp_bs_dbw_sky = eirp_bs_dbm_sky - 30.0
	
	eirp_ss_taxi_dbw = (30.0 - bw_loss_db) - 30.0
	eirp_ss_park_dbw = (26.0 - bw_loss_db) - 30.0
	
	p_rx_bs = eirp_bs_dbw_sky - fspl_db - l_atm + g_sat
	p_rx_ss_taxi = np.full_like(theta_deg, eirp_ss_taxi_dbw) - fspl_db - l_atm + g_sat
	p_rx_ss_park = np.full_like(theta_deg, eirp_ss_park_dbw) - fspl_db - l_atm + g_sat
	
	n_sys_w = k_boltz * t_sys * b_sat
	n_sys_dbw = 10.0 * np.log10(n_sys_w)
	
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
	
	ax1.plot(theta_deg, p_rx_bs, color='tab:blue', linewidth=2.3, label='Base station (BS) (EIRP 39.4 dBm + tilted antenna)')
	ax1.plot(theta_deg, p_rx_ss_taxi, color='tab:orange', linewidth=2.3, label='Aircraft (SS) - Taxiing (EIRP 30 dBm)')
	ax1.plot(theta_deg, p_rx_ss_park, color='tab:green', linewidth=2.3, label='Aircraft (SS) - Parking (EIRP 26 dBm)')
	ax1.axhline(target_p_rx_dbw, color='tab:red', linestyle='--', linewidth=2.0, label=f'Maximum satellite interference threshold ({target_p_rx_dbw} dBW)')
	
	ax1.set_title("AeroMACS interference power received at the Globalstar satellite")
	ax1.set_xlabel("Satellite elevation angle (degrees)")
	ax1.set_ylabel("Received power in 1.23 MHz bandwidth (dBW)")
	ax1.set_xlim(5, 90)
	ax1.set_ylim(-210, -140)
	ax1.grid(True, alpha=0.3)
	ax1.legend(loc="lower right", fontsize=8.5)
	
	i_dbw = np.linspace(-170.0, -150.0, 100)
	i_w = 10.0 ** (i_dbw / 10.0)
	temp_increase_percent = (i_w / n_sys_w) * 100.0
	
	ax2.plot(i_dbw, temp_increase_percent, color='tab:purple', linewidth=2.3, label='Interference temperature increase (%)')
	ax2.axvline(target_p_rx_dbw, color='tab:red', linestyle='--', linewidth=1.8, label=f'Threshold -157.3 dBW')
	i_target_w = 10.0 ** (target_p_rx_dbw / 10.0)
	target_increase = (i_target_w / n_sys_w) * 100.0
	ax2.scatter([target_p_rx_dbw], [target_increase], color='tab:red', s=60, zorder=5)
	ax2.text(target_p_rx_dbw + 0.5, target_increase + 0.1, f'{target_increase:.2f}% (Limit < 2.00%)', color='tab:red', fontweight='bold', fontsize=9)
	
	ax2.axhline(2.0, color='black', linestyle=':', linewidth=1.5, label='2% increase limit')
	
	ax2.set_title("Relationship between interference power and satellite noise temperature")
	ax2.set_xlabel("Integrated total interference power over 1.23 MHz (dBW)")
	ax2.set_ylabel("System noise temperature increase ratio ΔT / Tsys (%)")
	ax2.set_xlim(-170, -150)
	ax2.set_ylim(0, 5)
	ax2.grid(True, alpha=0.3)
	ax2.legend(loc="upper left", fontsize=9)
	
	plt.suptitle("Figure 10: Analysis of Globalstar MSS satellite protection criteria from the AeroMACS system at TSN", fontsize=13, fontweight='bold')
	fig.tight_layout()
	
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig10_mss_protection.png"
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def plot_fig11_received_psd_and_flatness(scenarios: dict[str, ScenarioConfig], output_dir: Path, show_plots: bool) -> Path:
	"""Figure 11: Received power spectral density (PSD) and spectral flatness at Tan Son Nhat"""
	nactive = 384
	bandwidth_mhz = 5.0
	df_khz = (bandwidth_mhz * 1e3) / 512.0
	
	parking_scenario = scenarios["parking"]
	taxiing_scenario = scenarios["taxiing"]
	
	d_link = 500.0
	f_hz = 5095e6
	c = 3e8
	fspl_1m = 20.0 * log10(4.0 * np.pi * 1.0 * f_hz / c)
	
	np.random.seed(101)
	shadow_parking = np.random.normal(0, parking_scenario.shadow_sigma_db)
	shadow_taxiing = np.random.normal(0, taxiing_scenario.shadow_sigma_db)
	
	pl_parking = fspl_1m + 10.0 * parking_scenario.eta * log10(d_link) + shadow_parking
	pl_taxiing = fspl_1m + 10.0 * taxiing_scenario.eta * log10(d_link) + shadow_taxiing
	
	tx_eirp_dbm = 39.4
	tx_eirp_subcarrier_dbm = tx_eirp_dbm - 10.0 * log10(nactive)
	
	taps_parking_us = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
	taps_taxiing_us = np.array([0.0, 0.25, 0.5, 1.0, 1.5])
	
	p_profile_parking = np.exp(-taps_parking_us / parking_scenario.delay_spread_us)
	p_profile_parking /= np.sum(p_profile_parking)
	
	p_profile_taxiing = np.exp(-taps_taxiing_us / taxiing_scenario.delay_spread_us)
	p_profile_taxiing /= np.sum(p_profile_taxiing)
	
	k_park_lin = 10.0 ** (parking_scenario.k_factor_db / 10.0)
	k_taxi_lin = 10.0 ** (taxiing_scenario.k_factor_db / 10.0)
	
	def generate_channel_freq_response(p_profile, taps_us, k_lin):
		los_power = k_lin / (k_lin + 1.0) * p_profile[0]
		nlos_power_0 = 1.0 / (k_lin + 1.0) * p_profile[0]
		
		h_taps = []
		h_0 = sqrt(los_power) + sqrt(nlos_power_0) * (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / sqrt(2)
		h_taps.append(h_0)
		
		for p in p_profile[1:]:
			h_tap = sqrt(p) * (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / sqrt(2)
			h_taps.append(h_tap)
			
		h_freq = np.zeros(nactive, dtype=complex)
		freqs_mhz = (np.arange(nactive) - nactive//2) * (df_khz / 1e3)
		for m in range(nactive):
			f = freqs_mhz[m] * 1e6
			val = 0.0
			for idx, tau in enumerate(taps_us):
				val += h_taps[idx] * np.exp(-1j * 2.0 * np.pi * f * (tau * 1e-6))
			h_freq[m] = val
		return h_freq
		
	h_freq_parking = generate_channel_freq_response(p_profile_parking, taps_parking_us, k_park_lin)
	h_freq_taxiing = generate_channel_freq_response(p_profile_taxiing, taps_taxiing_us, k_taxi_lin)
	
	rx_psd_parking = tx_eirp_subcarrier_dbm - pl_parking + 10.0 * np.log10(np.abs(h_freq_parking)**2)
	rx_psd_taxiing = tx_eirp_subcarrier_dbm - pl_taxiing + 10.0 * np.log10(np.abs(h_freq_taxiing)**2)
	
	num_realizations = 200
	diffs_parking = []
	diffs_taxiing = []
	for _ in range(num_realizations):
		h_p = generate_channel_freq_response(p_profile_parking, taps_parking_us, k_park_lin)
		h_t = generate_channel_freq_response(p_profile_taxiing, taps_taxiing_us, k_taxi_lin)
		
		p_p_db = 10.0 * np.log10(np.abs(h_p)**2)
		p_t_db = 10.0 * np.log10(np.abs(h_t)**2)
		
		diffs_parking.extend(np.abs(np.diff(p_p_db)))
		diffs_taxiing.extend(np.abs(np.diff(p_t_db)))
		
	diffs_parking = np.array(diffs_parking)
	diffs_taxiing = np.array(diffs_taxiing)
	
	flat_limit = 0.4
	comp_parking = np.mean(diffs_parking <= flat_limit) * 100.0
	comp_taxiing = np.mean(diffs_taxiing <= flat_limit) * 100.0
	
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
	
	subcarriers = np.arange(-nactive//2, nactive//2)
	ax1.plot(subcarriers, rx_psd_parking, color='tab:red', alpha=0.85, linewidth=1.4, label='Parking (NLOS effect, n=4)')
	ax1.plot(subcarriers, rx_psd_taxiing, color='tab:blue', alpha=0.85, linewidth=1.4, label='Taxiing (LOS effect, n=3)')
	
	ax1.axhline(np.mean(rx_psd_parking), color='darkred', linestyle='--', linewidth=1.8, label=f'Parking mean: {np.mean(rx_psd_parking):.1f} dBm')
	ax1.axhline(np.mean(rx_psd_taxiing), color='darkblue', linestyle='--', linewidth=1.8, label=f'Taxiing mean: {np.mean(rx_psd_taxiing):.1f} dBm')
	
	ax1.set_title("Received power spectral density (PSD) over OFDMA subcarriers")
	ax1.set_xlabel("Subcarrier index")
	ax1.set_ylabel("Received power per subcarrier (dBm)")
	ax1.grid(True, alpha=0.3)
	ax1.legend(loc="lower right", fontsize=8.5)
	
	bins = np.linspace(0, 2.0, 50)
	ax2.hist(diffs_parking, bins=bins, color='tab:red', alpha=0.55, edgecolor='darkred', density=True, label=f'Parking ({comp_parking:.1f}% <= 0.4 dB)')
	ax2.hist(diffs_taxiing, bins=bins, color='tab:blue', alpha=0.55, edgecolor='darkblue', density=True, label=f'Taxiing ({comp_taxiing:.1f}% <= 0.4 dB)')
	ax2.axvline(flat_limit, color='black', linestyle='--', linewidth=2.3)
	ax2.text(flat_limit + 0.05, 1.5, f"Spectral flatness limit\n{flat_limit} dB", color='black', fontweight='bold', fontsize=9.5)
	
	ax2.set_title("Distribution of power differences between adjacent subcarriers")
	ax2.set_xlabel("Power difference between adjacent subcarriers (dB)")
	ax2.set_ylabel("Distribution density")
	ax2.set_xlim(0, 2.0)
	ax2.grid(True, alpha=0.3)
	ax2.legend(loc="upper right", fontsize=9.5)
	
	plt.suptitle(f"Figure 11: Evaluation of received PSD and spectral flatness criteria at TSN", fontsize=13, fontweight='bold')
	fig.tight_layout()
	
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig11_spectral_flatness.png"
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="AeroMACS AMC simulation - Focusing purely on BER vs SNR")
	parser.add_argument("--headless", action="store_true", help="Do not open plot windows")
	parser.add_argument("--k-parking", type=float, default=-3.0, help="Rice factor K for parking scenario (dB)")
	parser.add_argument("--k-taxiing", type=float, default=5.0, help="Rice factor K for taxiing scenario (dB)")
	parser.add_argument("--output-dir", type=str, help="Directory to save output plots")
	return parser.parse_args()


def print_table(mcs_table: list[MCSConfig]) -> None:
	print("MCS table")
	print("name\t\tmod\tcode\trep\tthreshold_dB\tthroughput_Mbps")
	for mcs in mcs_table:
		print(
			f"{mcs.name}\t{mcs.modulation_order}\t{mcs.code_rate:.2f}\t{mcs.repetition}\t"
			f"{mcs.snr_threshold_db:>7.1f}\t\t{throughput_mbps(mcs):.3f}"
		)


def main() -> None:
	args = parse_args()
	mcs_table = build_mcs_table()
	scenarios = build_scenarios(k_parking=args.k_parking, k_taxiing=args.k_taxiing)
	if args.output_dir:
		output_dir = Path(args.output_dir)
	else:
		output_dir = Path(__file__).resolve().parent.parent / "simulation_results"
	show_plots = not args.headless and os.environ.get("AMC_HEADLESS") != "1"
	if not show_plots:
		plt.switch_backend("Agg")

	print_table(mcs_table)
	print("\nStarting the AeroMACS BER performance simulation...")

	# Figure 1: AWGN Theory
	fig1 = plot_fig1_awgn_theory(output_dir, show_plots)
	print(f"1. Saved AWGN theoretical BER plot (Figure 1): {fig1}")

	# Figure 2: AWGN Theory vs Sim
	fig2 = plot_fig2_awgn_simulation(output_dir, show_plots)
	print(f"2. Saved AWGN BER theory vs. simulation plot (Figure 2): {fig2}")

	# Figure 3: Parking BER
	fig3 = plot_fig3_parking_ber(scenarios["parking"], output_dir, show_plots)
	print(f"3. Saved parking scenario BER plot (Figure 3): {fig3}")

	# Figure 4: Taxiing BER
	fig4 = plot_fig4_taxiing_ber(scenarios["taxiing"], output_dir, show_plots)
	print(f"4. Saved taxiing scenario BER plot (Figure 4): {fig4}")

	# Figure 5: AMC Comparison
	fig5 = plot_fig5_amc_comparison(scenarios, mcs_table, output_dir, show_plots)
	print(f"5. Saved AMC performance comparison plot (Figure 5): {fig5}")

	# Figure 6: Throughput vs SNR
	fig6 = plot_fig6_throughput_vs_snr(scenarios, mcs_table, output_dir, show_plots)
	print(f"6. Saved throughput comparison plot (Figure 6): {fig6}")

	# Figure 7: Goodput vs SNR
	fig7 = plot_fig7_goodput_vs_snr(scenarios, mcs_table, output_dir, show_plots)
	print(f"7. Saved goodput comparison plot (Figure 7): {fig7}")

	# Figure 8: Time-series Flight Path Simulation
	fig8 = plot_fig8_timeseries_sim(scenarios, mcs_table, output_dir, show_plots, frames=200)
	print(f"8. Saved dynamic time-series simulation plot (Figure 8): {fig8}")

	# Figure 9: PSD Emission Mask M Simulation
	fig9 = plot_fig9_psd_emission_mask(output_dir, show_plots)
	print(f"9. Saved PSD emission mask simulation plot (Figure 9): {fig9}")

	# Figure 10: MSS (Globalstar) Satellite Protection
	fig10 = plot_fig10_mss_protection(output_dir, show_plots)
	print(f"10. Saved MSS protection simulation plot (Figure 10): {fig10}")

	# Figure 11: Fading Channel Received PSD and Flatness
	fig11 = plot_fig11_received_psd_and_flatness(scenarios, output_dir, show_plots)
	print(f"11. Saved received PSD flatness simulation plot (Figure 11): {fig11}")

	print("\nSimulation completed successfully!")
	print(f"All result images have been saved in the directory: {output_dir}")

	if show_plots:
		plt.show()


if __name__ == "__main__":
	main()