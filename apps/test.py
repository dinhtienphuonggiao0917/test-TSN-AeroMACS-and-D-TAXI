from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
import sys
from math import atan2, cos, erfc, log10, radians, sin, sqrt
from pathlib import Path
import argparse
import json
import csv

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

RESULT_DIAGRAM_LINE_WIDTH_SCALE = 1.5
RESULT_DIAGRAM_LEGEND_FONT_SCALE = 1.0

# Curves are intentionally distinguished by color only; no marker symbols are used.
RESULT_MARKERS = []

# Marker-related settings are kept for compatibility, but are disabled in the
# final result diagrams to avoid symbol clutter.
RESULT_MARKER_EVERY = None
RESULT_MARKER_SIZE = 0

# Result-diagram styling constants.
# Keep the default line color fallback while removing all marker emphasis.
RESULT_DIAGRAM_LINE_COLOR = "black"
RESULT_DIAGRAM_MARKER_FACE_COLOR = None
RESULT_DIAGRAM_MARKER_EDGE_COLOR = None
RESULT_DIAGRAM_MARKER_EDGE_WIDTH = 0.0
# Larger tick labels (22 is easier to read on big screens)
RESULT_DIAGRAM_TICK_FONT_SIZE = 22
RESULT_CURVE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
RESULT_MARKER_SYMBOLS = ["o", "s", "^", "D", "v", "P"]


plt.rcParams.update({
	"font.family": ["Times New Roman", "serif"],
	"font.size": 16,
	"axes.titlesize": 20,
	"axes.labelsize": 18,
	"legend.fontsize": 14,
	"xtick.labelsize": RESULT_DIAGRAM_TICK_FONT_SIZE,
	"ytick.labelsize": RESULT_DIAGRAM_TICK_FONT_SIZE,
	"axes.linewidth": 1.5,
	"lines.linewidth": 2.0,
	"lines.markersize": RESULT_MARKER_SIZE,
})

# Reconfigure stdout to use UTF-8 to prevent Unicode encoding errors on Windows
if hasattr(sys.stdout, "reconfigure"):
	sys.stdout.reconfigure(encoding="utf-8")

TARGET_BER = 1e-6
CHANNEL_BANDWIDTH_MHZ = 5.0
PEAK_THROUGHPUT_MBPS = 7.5
SAMPLES_PER_POINT = 2000


@dataclass(frozen=True)
class TSNBaseStation:
	station_id: str
	name: str
	latitude: float
	longitude: float
	line_of_sight: bool
	height_agl_m: float


@dataclass(frozen=True)
class TSNTestPoint:
	point_id: str
	latitude: float
	longitude: float
	height_agl_m: float


@dataclass(frozen=True)
class TSNLinkBudgetConfig:
	frequency_hz: float
	bandwidth_hz: float
	tx_eirp_dbm: float
	aircraft_antenna_gain_dbi: float
	receiver_noise_figure_db: float
	implementation_loss_db: float
	path_loss_exponent: float
	reference_distance_m: float
	obstacle_loss_db: float


def haversine_distance(
	latitude1: float,
	longitude1: float,
	latitude2: float,
	longitude2: float,
) -> float:
	"""Return the great-circle distance in metres between two WGS84 coordinates."""
	earth_radius_m = 6371000.0
	lat1 = radians(latitude1)
	lon1 = radians(longitude1)
	lat2 = radians(latitude2)
	lon2 = radians(longitude2)

	dlat = lat2 - lat1
	dlon = lon2 - lon1
	a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
	c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
	return earth_radius_m * c


def calculate_tsn_path_loss(
	distance_m: float,
	frequency_hz: float,
	path_loss_exponent: float,
	reference_distance_m: float,
	obstacle_loss_db: float,
) -> dict[str, float]:
	"""Calculate TSN link path loss without substituting unprovided calibration data."""
	if distance_m <= 0.0:
		raise ValueError("distance_m must be > 0")
	if frequency_hz <= 0.0:
		raise ValueError("frequency_hz must be > 0")
	if path_loss_exponent <= 0.0:
		raise ValueError("path_loss_exponent must be > 0")
	if reference_distance_m <= 0.0:
		raise ValueError("reference_distance_m must be > 0")
	if distance_m < reference_distance_m:
		raise ValueError("distance_m must be >= reference_distance_m")

	c = 299792458.0
	fspl_reference_db = 20.0 * log10(4.0 * 3.141592653589793 * reference_distance_m * frequency_hz / c)
	excess_loss_db = 10.0 * path_loss_exponent * log10(distance_m / reference_distance_m)
	path_loss_db = fspl_reference_db + excess_loss_db + obstacle_loss_db
	return {
		"free_space_reference_db": fspl_reference_db,
		"distance_excess_loss_db": excess_loss_db,
		"obstacle_loss_db": obstacle_loss_db,
		"path_loss_db": path_loss_db,
	}


def calculate_tsn_link_budget(
	distance_m: float,
	config: TSNLinkBudgetConfig,
) -> dict[str, float]:
	"""Calculate median received power and thermal-noise-limited SNR."""
	path_loss = calculate_tsn_path_loss(
		distance_m=distance_m,
		frequency_hz=config.frequency_hz,
		path_loss_exponent=config.path_loss_exponent,
		reference_distance_m=config.reference_distance_m,
		obstacle_loss_db=config.obstacle_loss_db,
	)
	received_power_dbm = (
		config.tx_eirp_dbm
		+ config.aircraft_antenna_gain_dbi
		- path_loss["path_loss_db"]
	)
	noise_floor_dbm = (
		-174.0
		+ 10.0 * log10(config.bandwidth_hz)
		+ config.receiver_noise_figure_db
	)
	snr_db = received_power_dbm - noise_floor_dbm - config.implementation_loss_db
	return {
		**path_loss,
		"received_power_dbm": received_power_dbm,
		"noise_floor_dbm": noise_floor_dbm,
		"snr_db": snr_db,
	}


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


class MCSStatus(Enum):
	SERVICEABLE = "serviceable"
	NOT_SERVICEABLE = "not_serviceable"


@dataclass(frozen=True)
class MCSSelection:
	status: MCSStatus
	selected_mcs: MCSConfig | None

	def __post_init__(self) -> None:
		if (self.status is MCSStatus.SERVICEABLE) != (self.selected_mcs is not None):
			raise ValueError("MCS status must agree with whether a configuration was selected")


@dataclass(frozen=True)
class AMCBerEstimate:
	conditional_ber: np.ndarray
	serviceability: np.ndarray


@dataclass(frozen=True)
class BaseStationLayoutConfig:
	name: str
	base_station_count: int
	los_station_count: int
	closer_nlos_count: int
	avg_inter_bs_distance_km: float
	coverage_overlap: float
	traffic_load_factor: float
	bs_snr_gain_db: float


@dataclass(frozen=True)
class ScenarioConfig:
	name: str
	mean_snr_db: float
	k_factor_db: float
	eta: float
	shadow_sigma_db: float
	fast_fading_sigma_db: float
	deep_fade_prob: float
	deep_fade_penalty_db: float
	ar_coeff: float
	doppler_hz: float
	delay_spread_us: float
	base_station_count: int = 8
	bs_layout_name: str = "TSN_8BS"
	bs_los_count: int = 4
	bs_nlos_count: int = 4
	bs_snr_gain_db: float = 1.5
	avg_inter_bs_distance_km: float = 1.52
	coverage_overlap: float = 0.82
	traffic_load_factor: float = 1.0


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


def build_tsn_bs_layout() -> list[TSNBaseStation]:
	"""Geographic layout supplied for the 8 proposed TSN base stations (WGS84)."""
	return [
		TSNBaseStation("BS-1", "Mái nhà ga T1", 10.81142, 106.64380, False, 20.0),
		TSNBaseStation("BS-2", "Mái nhà ga T2", 10.81467, 106.65255, False, 20.0),
		TSNBaseStation("BS-3", "Mái nhà ga T3", 10.81830, 106.66199, False, 20.0),
		TSNBaseStation("BS-4", "Đường băng Nam (25R)", 10.81573, 106.63858, True, 7.0),
		TSNBaseStation("BS-5", "Đường băng Bắc (25L)", 10.82314, 106.65832, True, 7.0),
		TSNBaseStation("BS-6", "Trục đường lăn trung tâm", 10.81776, 106.65044, True, 7.0),
		TSNBaseStation("BS-7", "Khu hàng hóa (Cargo)", 10.81285, 106.65912, False, 20.0),
		TSNBaseStation("BS-8", "Khu ARFF và dịch vụ khẩn nguy", 10.81197, 106.66550, True, 20.0),
	]


def build_tsn_test_points() -> list[TSNTestPoint]:
	return [
		TSNTestPoint("P1", 10.8164166667, 106.6618611111, 17.0),
		TSNTestPoint("P2", 10.8165833333, 106.6592222222, 17.0),
		TSNTestPoint("P3", 10.8131250000, 106.6589083333, 17.0),
		TSNTestPoint("P4", 10.8131861111, 106.6580361111, 17.0),
		TSNTestPoint("P5", 10.8128611111, 106.6571111111, 17.0),
	]


def build_scenarios(
	k_parking: float = -3.0,
	k_taxiing: float = 5.0,
	base_station_count: int = 8,
	bs_layout_name: str = "TSN_8BS",
) -> dict[str, ScenarioConfig]:
	"""Build parking/taxiing scenario calibration points for the active BS layout.

	The uploaded station layout describes 8 strategically placed BSs with 4 LoS sites
	(BS-4, BS-5, BS-6, BS-8) and 4 NLoS sites (BS-1, BS-2, BS-3, BS-7), so the
	new simulation model optionally carries these layout parameters through the
	ScenarioConfig object instead of leaving them as narrative-only input.
	"""
	if base_station_count < 1:
		raise ValueError("base_station_count must be >= 1")

	los_count = min(4, max(0, base_station_count // 2))
	nlos_count = max(0, base_station_count - los_count)
	if base_station_count == 8:
		los_count = 4
		nlos_count = 4
	
	layout_gain_db = 0.35 * (base_station_count - 2)
	if base_station_count >= 8:
		layout_gain_db = 1.5
	
	avg_inter_bs_distance_km = 1.52
	if base_station_count != 8:
		avg_inter_bs_distance_km = max(0.5, 1.52 * (8 / base_station_count) ** 0.35)

	layout_cfg = BaseStationLayoutConfig(
		name=bs_layout_name,
		base_station_count=base_station_count,
		los_station_count=los_count,
		closer_nlos_count=nlos_count,
		avg_inter_bs_distance_km=avg_inter_bs_distance_km,
		coverage_overlap=0.82,
		traffic_load_factor=1.0,
		bs_snr_gain_db=layout_gain_db,
	)

	return {
		"parking": ScenarioConfig(
			name="Parking",
			mean_snr_db=8.0 + 0.1 * (base_station_count - 2),
			k_factor_db=k_parking,
			eta=4.0,
			shadow_sigma_db=8.0,
			fast_fading_sigma_db=1.6,
			deep_fade_prob=0.32,
			deep_fade_penalty_db=4.0,
			ar_coeff=0.70,
			doppler_hz=25.0,
			delay_spread_us=3.0,
			base_station_count=base_station_count,
			bs_layout_name=bs_layout_name,
			bs_los_count=los_count,
			bs_nlos_count=nlos_count,
			bs_snr_gain_db=layout_gain_db,
			avg_inter_bs_distance_km=avg_inter_bs_distance_km,
			coverage_overlap=layout_cfg.coverage_overlap,
			traffic_load_factor=layout_cfg.traffic_load_factor,
		),
		"taxiing": ScenarioConfig(
			name="Taxiing",
			mean_snr_db=15.5 + 0.1 * (base_station_count - 2),
			k_factor_db=k_taxiing,
			eta=3.0,
			shadow_sigma_db=1.2,
			fast_fading_sigma_db=1.0,
			deep_fade_prob=0.08,
			deep_fade_penalty_db=2.5,
			ar_coeff=0.55,
			doppler_hz=438.0,
			delay_spread_us=1.5,
			base_station_count=base_station_count,
			bs_layout_name=bs_layout_name,
			bs_los_count=los_count,
			bs_nlos_count=nlos_count,
			bs_snr_gain_db=layout_gain_db,
			avg_inter_bs_distance_km=avg_inter_bs_distance_km,
			coverage_overlap=layout_cfg.coverage_overlap,
			traffic_load_factor=layout_cfg.traffic_load_factor,
		),
	}


def select_mcs(snr_db: float, mcs_table: list[MCSConfig]) -> MCSSelection:
	if not mcs_table:
		return MCSSelection(MCSStatus.NOT_SERVICEABLE, None)
	minimum_threshold_db = min(mcs.snr_threshold_db for mcs in mcs_table)
	if snr_db < minimum_threshold_db:
		return MCSSelection(MCSStatus.NOT_SERVICEABLE, None)
	selected = max(
		(mcs for mcs in mcs_table if snr_db >= mcs.snr_threshold_db),
		key=lambda mcs: mcs.snr_threshold_db,
		default=None,
	)
	if selected is None:
		return MCSSelection(MCSStatus.NOT_SERVICEABLE, None)
	return MCSSelection(MCSStatus.SERVICEABLE, selected)


def effective_ebn0_db(snr_db: float, mcs: MCSConfig) -> float:
	spectral_efficiency = mcs.bits_per_symbol * mcs.code_rate / mcs.repetition
	return snr_db - 10.0 * log10(spectral_efficiency)


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
	base_snr = scenario.mean_snr_db + 0.35 * scenario.k_factor_db - 1.2 * (scenario.eta - 3.0)
	prev_delta = 0.0
	doppler_scale = min(1.0, scenario.doppler_hz / 200.0)
	for index in range(frames):
		slow_shadow = rng.normal(0.0, scenario.shadow_sigma_db)
		fast_fading = rng.normal(0.0, scenario.fast_fading_sigma_db * (1.0 + 0.5 * doppler_scale))
		deep_fade = -scenario.deep_fade_penalty_db if rng.random() < scenario.deep_fade_prob else 0.0
		prev_delta = scenario.ar_coeff * prev_delta + (1.0 - scenario.ar_coeff) * (slow_shadow + fast_fading + deep_fade)
		snr_series[index] = np.clip(base_snr + prev_delta, -5.0, 30.0)
	return snr_series


def simulate_per_per_bs(
	scenario: ScenarioConfig,
	mcs_table: list[MCSConfig],
	frames: int,
	packet_size_bits: int,
	rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Simulate SNR, BER and PER time-series per base-station for the scenario.
	Returns SNR, BER, PER and outage matrices with shape (bs, frames).
	BER is NaN during outage because no MCS is available; PER is 1.0 because
	the packet is not served.

	This uses the scenario.bs_los_count and scenario.bs_snr_gain_db to
	create simple per-BS mean SNR offsets (LOS get +bs_snr_gain_db, NLOS get a
	half-penalty). The channel series generator is reused for temporal
	variations.
	"""
	bs = max(1, int(getattr(scenario, "base_station_count", 1)))
	los = int(getattr(scenario, "bs_los_count", bs // 2))
	# Preallocate matrices
	snr_matrix = np.empty((bs, frames), dtype=float)
	ber_matrix = np.empty((bs, frames), dtype=float)
	per_matrix = np.empty((bs, frames), dtype=float)
	outage_matrix = np.zeros((bs, frames), dtype=bool)

	for i in range(bs):
		# Simple LOS/NLOS mean SNR adjustment
		if i < los:
			mean_snr = scenario.mean_snr_db + float(getattr(scenario, "bs_snr_gain_db", 0.0))
		else:
			# NLOS penalty (half of gain) to spread BS mean levels
			mean_snr = scenario.mean_snr_db - max(0.0, float(getattr(scenario, "bs_snr_gain_db", 0.0)) * 0.5)

		# Create a per-BS ScenarioConfig with adjusted mean
		sc = ScenarioConfig(
			name=f"{scenario.name}_BS{i+1}",
			mean_snr_db=mean_snr,
			k_factor_db=scenario.k_factor_db,
			eta=scenario.eta,
			shadow_sigma_db=scenario.shadow_sigma_db,
			fast_fading_sigma_db=scenario.fast_fading_sigma_db,
			deep_fade_prob=scenario.deep_fade_prob,
			deep_fade_penalty_db=scenario.deep_fade_penalty_db,
			ar_coeff=scenario.ar_coeff,
			doppler_hz=scenario.doppler_hz,
			delay_spread_us=scenario.delay_spread_us,
			base_station_count=1,
		)

		# Generate SNR time-series for this BS
		snr_series = simulate_channel_series(sc, frames, rng)
		snr_matrix[i, :] = snr_series

		# Convert to BER / PER per-frame
		for j, x in enumerate(snr_series):
			selection = select_mcs(float(x), mcs_table)
			selected = selection.selected_mcs
			if selection.status is MCSStatus.NOT_SERVICEABLE or selected is None:
				ber_matrix[i, j] = np.nan
				per_matrix[i, j] = 1.0
				outage_matrix[i, j] = True
				continue
			ber = ber_for_mcs(float(x), selected)
			per = 1.0 - (1.0 - ber) ** packet_size_bits
			ber_matrix[i, j] = ber
			per_matrix[i, j] = per

	return snr_matrix, ber_matrix, per_matrix, outage_matrix


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
			mean_snr_db=snr_db,
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
) -> AMCBerEstimate:
	"""Estimate BER over served frames and serviceability over the SNR range."""
	rng = np.random.default_rng(seed)
	ber_curve = np.full_like(snr_range_db, np.nan, dtype=float)
	serviceability_curve = np.zeros_like(snr_range_db, dtype=float)
	for index, snr_db in enumerate(snr_range_db):
		scenario_point = ScenarioConfig(
			name=scenario.name,
			mean_snr_db=snr_db,
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
			selection = select_mcs(float(x) - esp_loss_db, mcs_table)
			selected = selection.selected_mcs
			if selection.status is MCSStatus.NOT_SERVICEABLE or selected is None:
				continue
			# Actual transmission BER is computed at estimated SNR (reflecting estimation loss)
			ber_val = ber_for_mcs(float(x) - esp_loss_db, selected)
			ber_list.append(ber_val)
		serviceability_curve[index] = len(ber_list) / len(snr_series)
		ber_curve[index] = float(np.mean(ber_list)) if ber_list else np.nan
	return AMCBerEstimate(ber_curve, serviceability_curve)


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
			mean_snr_db=snr_db,
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
			mean_snr_db=snr_db,
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
			selection = select_mcs(float(x) - esp_loss_db, mcs_table)
			selected = selection.selected_mcs
			if selection.status is MCSStatus.NOT_SERVICEABLE or selected is None:
				tp_list.append(0.0)
				gp_list.append(0.0)
				continue
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


def apply_result_diagram_style(fig: plt.Figure) -> None:
	"""Scale saved result-figure line work and legend fonts for presentation."""
	for ax in fig.axes:
		ax.set_facecolor("#FAFCFE")
		ax.grid(True, which="major", color="#B8C6D1", alpha=0.55, linewidth=0.8)
		ax.grid(True, which="minor", color="#DCE5EA", alpha=0.35, linewidth=0.5)
		for spine in ax.spines.values():
			spine.set_color("#526773")
		# Enforce the requested visible result-diagram presentation.
		ax.tick_params(axis="x", labelsize=RESULT_DIAGRAM_TICK_FONT_SIZE)
		ax.tick_params(axis="y", labelsize=RESULT_DIAGRAM_TICK_FONT_SIZE)

		# Only color is used to distinguish curves; no marker symbols are drawn.
		for line in ax.lines:
			try:
				line_color = line.get_color()
				linewidth = line.get_linewidth()
				if linewidth is not None:
					line.set_linewidth(linewidth * RESULT_DIAGRAM_LINE_WIDTH_SCALE)
				line.set_marker("")
				line.set_markevery(None)
				line.set_markersize(0)
				if line_color is None:
					line.set_color(RESULT_DIAGRAM_LINE_COLOR)
			except Exception:
				pass

		legend = ax.get_legend()
		if legend is not None:
			# Grow legend entry text size.
			texts = legend.get_texts()
			if texts:
				for text in texts:
					try:
						current_size = text.get_fontsize()
						if current_size is not None:
							text.set_fontsize(current_size * RESULT_DIAGRAM_LEGEND_FONT_SCALE)
					except Exception:
						pass

			# Keep legend handles line-only; markers are intentionally suppressed.
			try:
				legend_handles = legend.legend_handles
				if legend_handles:
					for handle in legend_handles:
						try:
							handle_color = handle.get_color()
							current_width = handle.get_linewidth()
							if current_width is not None:
								handle.set_linewidth(current_width * RESULT_DIAGRAM_LINE_WIDTH_SCALE)
							handle.set_marker("")
							handle.set_markersize(0)
							if handle_color is None:
								handle.set_color(RESULT_DIAGRAM_LINE_COLOR)
						except Exception:
							pass
			except Exception:
				pass


# ---------------------------------------------------------
# NEW PLOTTING FUNCTIONS
# ---------------------------------------------------------

def plot_fig1_awgn_theory(output_dir: Path, show_plots: bool) -> Path:
	"""Figure 1: Theoretical BER limit in AWGN channel"""
	snr_range_db = np.arange(-5.0, 25.5, 0.5)
	mcs_table = build_mcs_table()
	colors = RESULT_CURVE_COLORS
	
	fig, ax = plt.subplots(figsize=(10, 6))
	for mcs, color in zip(mcs_table, colors):
		ax.semilogy(snr_range_db, awgn_ber_curve(mcs, snr_range_db), label=mcs.name, color=color, linewidth=2.3)
		
	ax.axhline(TARGET_BER, color="black", linestyle="--", linewidth=1.2, label=f"Target BER = {TARGET_BER:.0e}")
	ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.2, label="Minimum MCS threshold (0 dB)")
	ax.set_title("Figure 1: Theoretical BER Limit of MCS Schemes in AWGN Channel")
	ax.set_xlabel("Signal-to-Noise Ratio (SNR) (dB)")
	ax.set_ylabel("Bit Error Rate (BER)")
	ax.set_ylim(1e-12, 5e-1)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	ax.legend(loc="upper right", fontsize=9)
	
	apply_result_diagram_style(fig)
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig1_awgn_theory.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def plot_fig2_awgn_simulation(output_dir: Path, show_plots: bool) -> Path:
	"""Figure 2: BER performance in AWGN channel (theory vs. simulation)"""
	snr_theory_db = np.arange(-5.0, 25.5, 0.5)
	snr_sim_db = np.concatenate((np.arange(-5.0, 0.0, 2.0), np.arange(0.0, 26.0, 2.0)))
	mcs_selections = [
		("QPSK_1_2", "QPSK 1/2"),
		("16QAM_1_2", "16-QAM 1/2"),
		("64QAM_3_4", "64-QAM 3/4"),
	]
	mcs_table = {mcs.name: mcs for mcs in build_mcs_table()}
	
	fig, ax = plt.subplots(figsize=(12, 7.5))
	for index, (mcs_name, label_name) in enumerate(mcs_selections):
		mcs = mcs_table[mcs_name]
		color = RESULT_CURVE_COLORS[index]
		marker = RESULT_MARKER_SYMBOLS[index]
		theory_ber = awgn_ber_curve(mcs, snr_theory_db)
		sim_points = np.array([
			simulate_awgn_ber(mcs, float(snr_db), seed=42 + index * 100 + point_index)
			for point_index, snr_db in enumerate(snr_sim_db)
		])
		ax.semilogy(snr_theory_db, theory_ber, color=color, linestyle='-', linewidth=2.3, label=f"Theory {label_name}")
		# Filter out zero values for correct log plot representation
		sim_points = np.asarray(sim_points)
		valid_idx = sim_points > 0
		ax.scatter(
			snr_sim_db[valid_idx],
			sim_points[valid_idx],
			color=color,
			facecolors="white",
			edgecolors=color,
			marker=marker,
			s=42,
			linewidths=1.0,
			label=f"Simulation {label_name}",
			zorder=3,
		)
		
	ax.axhline(TARGET_BER, color="black", linestyle="--", linewidth=1.2, label=f"Target BER = {TARGET_BER:.0e}")
	ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.2, label="Minimum MCS threshold (0 dB)")
	ax.set_title("Figure 2: BER Performance in AWGN Channel (Theory vs. Monte Carlo Simulation)")
	ax.set_xlabel("Signal-to-Noise Ratio (SNR) (dB)")
	ax.set_ylabel("Bit Error Rate (BER)")
	ax.set_ylim(1e-7, 5e-1)
	ax.set_xlim(snr_theory_db[0], snr_theory_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	ax.legend(loc="upper right", fontsize=9)
	
	apply_result_diagram_style(fig)
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
	
	colors = RESULT_CURVE_COLORS[:5]
	
	fig, ax = plt.subplots(figsize=(11, 7))
	
	for index, (mcs, color) in enumerate(zip(mcs_list, colors)):
		awgn_curve = awgn_ber_curve(mcs, snr_range_db)
		pck_curve = estimate_scenario_ber_curve(
			scenario, mcs, snr_range_db, SAMPLES_PER_POINT, seed=100 + index
		)
		esp_curve = estimate_scenario_ber_curve(
			scenario, mcs, snr_range_db, SAMPLES_PER_POINT, seed=100 + index, esp_loss_db=2.0
		)
		ax.semilogy(snr_range_db, awgn_curve, color=color, linestyle=':', linewidth=1.2)
		ax.semilogy(snr_range_db, pck_curve, color=color, linestyle='-', linewidth=2.0)
		ax.semilogy(snr_range_db, esp_curve, color=color, linestyle='--', linewidth=1.6)
	
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
	ax.set_xlabel("Signal-to-Noise Ratio (SNR) (dB)")
	ax.set_ylabel("Bit Error Rate (BER)")
	ax.set_ylim(1e-6, 5e-1)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	
	# Place the legend
	ax.legend(handles=color_handles + style_handles, loc="upper right", fontsize=8)
	
	apply_result_diagram_style(fig)
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig3_parking_ber_vs_snr.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def plot_fig4_taxiing_ber(scenario: ScenarioConfig, output_dir: Path, show_plots: bool) -> Path:
	"""Figure 4: BER performance in Taxiing scenario (LOS)"""
	snr_range_db = np.arange(-5.0, 25.5, 0.5)
	mcs_list = [
		MCSConfig("16QAM_1_2", 16, 0.5, 1, 10.5),
		MCSConfig("16QAM_3_4", 16, 0.75, 1, 13.4),
		MCSConfig("64QAM_3_4", 64, 0.75, 1, 17.8),
	]
	
	colors = RESULT_CURVE_COLORS[2:5]
	
	fig, ax = plt.subplots(figsize=(11, 7))
	
	for index, (mcs, color) in enumerate(zip(mcs_list, colors)):
		awgn_curve = awgn_ber_curve(mcs, snr_range_db)
		pck_curve = estimate_scenario_ber_curve(
			scenario, mcs, snr_range_db, SAMPLES_PER_POINT, seed=200 + index
		)
		esp_curve = estimate_scenario_ber_curve(
			scenario, mcs, snr_range_db, SAMPLES_PER_POINT, seed=200 + index, esp_loss_db=2.0
		)
		ax.semilogy(snr_range_db, awgn_curve, color=color, linestyle=':', linewidth=1.2)
		ax.semilogy(snr_range_db, pck_curve, color=color, linestyle='-', linewidth=2.0)
		ax.semilogy(snr_range_db, esp_curve, color=color, linestyle='--', linewidth=1.6)
		
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
	ax.set_xlabel("Signal-to-Noise Ratio (SNR) (dB)")
	ax.set_ylabel("Bit Error Rate (BER)")
	ax.set_ylim(1e-12, 5e-1)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	
	# Place the legend
	ax.legend(handles=color_handles + style_handles, loc="upper right", fontsize=8)
	
	apply_result_diagram_style(fig)
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig4_taxiing_ber_vs_snr.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def plot_mcs_serviceability_thresholds(
	mcs_table: list[MCSConfig],
	output_dir: Path,
	show_plots: bool,
) -> tuple[Path, Path]:
	"""Plot selected MCS and export serviceability checks around every threshold."""
	if not mcs_table:
		raise ValueError("mcs_table must contain at least one MCS")

	ordered_mcs = sorted(mcs_table, key=lambda mcs: mcs.snr_threshold_db)
	thresholds = sorted({mcs.snr_threshold_db for mcs in ordered_mcs})
	lower_snr_db = thresholds[0] - 1.0
	upper_snr_db = thresholds[-1] + 1.0
	mcs_levels = {mcs.name: index + 1 for index, mcs in enumerate(ordered_mcs)}

	def selection_level(snr_db: float) -> int:
		selection = select_mcs(snr_db, ordered_mcs)
		if selection.status is MCSStatus.NOT_SERVICEABLE or selection.selected_mcs is None:
			return 0
		return mcs_levels[selection.selected_mcs.name]

	edges = [lower_snr_db, *thresholds, upper_snr_db]
	levels = [
		selection_level((left + right) / 2.0)
		for left, right in zip(edges, edges[1:])
	]
	fig, ax = plt.subplots(figsize=(12, 7))
	minimum_threshold_db = thresholds[0]
	ax.axvspan(
		lower_snr_db,
		minimum_threshold_db,
		color="#D9E1E8",
		alpha=0.65,
		label=f"Not serviceable below {minimum_threshold_db:g} dB",
	)
	ax.stairs(levels, edges, color=RESULT_CURVE_COLORS[0], linewidth=2.5, label="Selected MCS/status")

	probe_offsets_db = (-0.1, 0.0, 0.1)
	probe_rows: list[dict[str, str]] = []
	for threshold_db in thresholds:
		ax.axvline(threshold_db, color="gray", linestyle=":", linewidth=1.0)
		for offset_db in probe_offsets_db:
			snr_db = threshold_db + offset_db
			selection = select_mcs(snr_db, ordered_mcs)
			selected_mcs = selection.selected_mcs
			probe_rows.append({
				"threshold_db": f"{threshold_db:.1f}",
				"offset_db": f"{offset_db:+.1f}",
				"snr_db": f"{snr_db:.1f}",
				"status": selection.status.value,
				"selected_mcs": selected_mcs.name if selected_mcs is not None else "NOT_SERVICEABLE",
			})
			ax.scatter(
				[snr_db],
				[selection_level(snr_db)],
				color="black",
				s=28,
				zorder=3,
			)

	ax.set_title("MCS Selection and Serviceability at Switching Thresholds")
	ax.set_xlabel("Instantaneous SNR (dB)")
	ax.set_ylabel("Selected transmission mode")
	ax.set_xlim(lower_snr_db, upper_snr_db)
	ax.set_xticks(thresholds, [f"{threshold:g}" for threshold in thresholds])
	ax.set_yticks(
		range(len(ordered_mcs) + 1),
		["NOT SERVICEABLE", *(mcs.name for mcs in ordered_mcs)],
	)
	ax.grid(True, axis="both", alpha=0.35)
	ax.legend(loc="upper left", fontsize=9)
	apply_result_diagram_style(fig)

	output_dir.mkdir(parents=True, exist_ok=True)
	plot_path = output_dir / "fig_mcs_selection_serviceability.png"
	csv_path = output_dir / "mcs_threshold_probe_results.csv"
	fig.tight_layout()
	fig.savefig(plot_path, dpi=160, bbox_inches="tight")
	with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(
			csv_file,
			fieldnames=["threshold_db", "offset_db", "snr_db", "status", "selected_mcs"],
		)
		writer.writeheader()
		writer.writerows(probe_rows)
	if not show_plots:
		plt.close(fig)
	return plot_path, csv_path


def plot_fig5_amc_comparison(scenarios: dict[str, ScenarioConfig], mcs_table: list[MCSConfig], output_dir: Path, show_plots: bool) -> Path:
	"""Figure 5: Comparison of BER performance of AMC and optimal MCS schemes"""
	snr_range_db = np.arange(-5.0, 25.5, 0.5)
	parking_scenario = scenarios["parking"]
	taxiing_scenario = scenarios["taxiing"]
	parking_fixed_mcs = next(mcs for mcs in mcs_table if mcs.name == "QPSK_1_2_REP2")
	taxiing_fixed_mcs = next(mcs for mcs in mcs_table if mcs.name == "64QAM_3_4")
	parking_fixed_pck = estimate_scenario_ber_curve(
		parking_scenario, parking_fixed_mcs, snr_range_db, SAMPLES_PER_POINT, seed=500
	)
	parking_fixed_esp = estimate_scenario_ber_curve(
		parking_scenario, parking_fixed_mcs, snr_range_db, SAMPLES_PER_POINT, seed=500, esp_loss_db=2.0
	)
	taxiing_fixed_pck = estimate_scenario_ber_curve(
		taxiing_scenario, taxiing_fixed_mcs, snr_range_db, SAMPLES_PER_POINT, seed=501
	)
	taxiing_fixed_esp = estimate_scenario_ber_curve(
		taxiing_scenario, taxiing_fixed_mcs, snr_range_db, SAMPLES_PER_POINT, seed=501, esp_loss_db=2.0
	)
	parking_amc_pck = estimate_amc_ber_curve(
		parking_scenario, mcs_table, snr_range_db, SAMPLES_PER_POINT, seed=502
	).conditional_ber
	parking_amc_esp = estimate_amc_ber_curve(
		parking_scenario, mcs_table, snr_range_db, SAMPLES_PER_POINT, seed=502, esp_loss_db=2.0
	).conditional_ber
	taxiing_amc_pck = estimate_amc_ber_curve(
		taxiing_scenario, mcs_table, snr_range_db, SAMPLES_PER_POINT, seed=503
	).conditional_ber
	taxiing_amc_esp = estimate_amc_ber_curve(
		taxiing_scenario, mcs_table, snr_range_db, SAMPLES_PER_POINT, seed=503, esp_loss_db=2.0
	).conditional_ber

	fig, ax = plt.subplots(figsize=(12, 8))
	
	# 1. Parking optimal fixed MCS (QPSK 1/2 REP2)
	ax.semilogy(snr_range_db, parking_fixed_pck, 'red', linestyle='-', alpha=0.5, linewidth=1.8, label='Parking: QPSK 1/2 REP2 (PCK)')
	ax.semilogy(snr_range_db, parking_fixed_esp, 'red', linestyle='--', alpha=0.5, linewidth=1.6, label='Parking: QPSK 1/2 REP2 (ESP)')
	
	# 2. Taxiing optimal fixed MCS (64-QAM 3/4)
	ax.semilogy(snr_range_db, taxiing_fixed_pck, 'blue', linestyle='-', alpha=0.5, linewidth=1.8, label='Taxiing: 64-QAM 3/4 (PCK)')
	ax.semilogy(snr_range_db, taxiing_fixed_esp, 'blue', linestyle='--', alpha=0.5, linewidth=1.6, label='Taxiing: 64-QAM 3/4 (ESP)')
	
	# 3. Dynamic AMC BER curves
	ax.semilogy(snr_range_db, parking_amc_pck, 'green', linestyle='-', linewidth=2.8, label='Parking: AMC BER (served frames, PCK)')
	ax.semilogy(snr_range_db, parking_amc_esp, 'green', linestyle='--', linewidth=2.3, label='Parking: AMC BER (served frames, ESP)')
	
	ax.semilogy(snr_range_db, taxiing_amc_pck, 'darkorange', linestyle='-', linewidth=2.8, label='Taxiing: AMC BER (served frames, PCK)')
	ax.semilogy(snr_range_db, taxiing_amc_esp, 'darkorange', linestyle='--', linewidth=2.3, label='Taxiing: AMC BER (served frames, ESP)')
	
	# Horizontal target line
	ax.axhline(TARGET_BER, color="black", linestyle="--", linewidth=1.6, label=f"Target BER = {TARGET_BER:.0e}")
	
	# Vertical lines indicating AMC switching thresholds
	minimum_threshold_db = min(mcs.snr_threshold_db for mcs in mcs_table)
	ax.axvspan(
		snr_range_db[0],
		minimum_threshold_db,
		color="#D9E1E8",
		alpha=0.45,
		label=f"Mean SNR below minimum MCS ({minimum_threshold_db:g} dB)",
	)
	thresholds = [mcs.snr_threshold_db for mcs in mcs_table if mcs.snr_threshold_db > minimum_threshold_db]
	thresholds = sorted(list(set(thresholds)))
	ax.axvline(minimum_threshold_db, color="gray", linestyle=":", alpha=0.8, label="Minimum service threshold")
	for t in thresholds:
		ax.axvline(t, color='gray', linestyle=':', alpha=0.6)
		ax.text(t + 0.2, 5e-12, f"{t} dB", rotation=90, verticalalignment='bottom', fontsize=8, color='dimgray')
		
	ax.set_title("Figure 5: Comparison of BER Performance of Adaptive AMC System and Fixed MCS Configurations")
	ax.set_xlabel("Signal-to-Noise Ratio (SNR) (dB)")
	ax.set_ylabel("Bit Error Rate (BER)")
	ax.set_ylim(1e-12, 5e-1)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	ax.legend(loc="upper right", fontsize=8.5)
	
	apply_result_diagram_style(fig)
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig5_amc_comparison.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	if not show_plots:
		plt.close(fig)
	return output_path


def plot_taxiing_amc_serviceability(
	scenario: ScenarioConfig,
	mcs_table: list[MCSConfig],
	output_dir: Path,
	show_plots: bool,
) -> tuple[Path, Path]:
	"""Plot Taxiing AMC BER on served frames and the fraction of serviceable frames."""
	if not mcs_table:
		raise ValueError("mcs_table must contain at least one MCS")

	snr_range_db = np.arange(-5.0, 25.5, 0.5)
	estimate = estimate_amc_ber_curve(
		scenario,
		mcs_table,
		snr_range_db,
		SAMPLES_PER_POINT,
		seed=503,
	)
	minimum_threshold_db = min(mcs.snr_threshold_db for mcs in mcs_table)
	thresholds = sorted({mcs.snr_threshold_db for mcs in mcs_table})

	fig, ax = plt.subplots(figsize=(12, 7))
	service_ax = ax.twinx()
	ax.axvspan(
		snr_range_db[0],
		minimum_threshold_db,
		color="#D9E1E8",
		alpha=0.45,
		label=f"Mean SNR below minimum MCS ({minimum_threshold_db:g} dB)",
	)
	ax.semilogy(
		snr_range_db,
		estimate.conditional_ber,
		color=RESULT_CURVE_COLORS[0],
		label="Conditional BER (served frames)",
	)
	service_ax.plot(
		snr_range_db,
		estimate.serviceability * 100.0,
		color=RESULT_CURVE_COLORS[1],
		linestyle="--",
		label="Serviceable frames",
	)
	for threshold_db in thresholds:
		ax.axvline(threshold_db, color="gray", linestyle=":", linewidth=0.8, alpha=0.65)

	ax.set_title(
		f"Taxiing AMC BER and Serviceability (K = {scenario.k_factor_db:.1f} dB)"
	)
	ax.set_xlabel("Mean SNR (dB)")
	ax.set_ylabel("BER when a transmission is served")
	service_ax.set_ylabel("Serviceable frames (%)")
	service_ax.set_ylim(0.0, 100.0)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.set_ylim(1e-12, 5e-1)
	ax.grid(True, which="both", alpha=0.35)
	handles, labels = ax.get_legend_handles_labels()
	service_handles, service_labels = service_ax.get_legend_handles_labels()
	ax.legend(handles + service_handles, labels + service_labels, loc="upper right", fontsize=9)
	apply_result_diagram_style(fig)

	output_dir.mkdir(parents=True, exist_ok=True)
	plot_path = output_dir / "fig_taxiing_amc_ber_serviceability.png"
	csv_path = output_dir / "taxiing_amc_serviceability.csv"
	fig.tight_layout()
	fig.savefig(plot_path, dpi=160, bbox_inches="tight")
	with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow([
			"mean_snr_db",
			"conditional_ber_when_served",
			"serviceable_fraction",
			"serviceable_percent",
		])
		for snr_db, ber, serviceability in zip(
			snr_range_db, estimate.conditional_ber, estimate.serviceability
		):
			writer.writerow([
				f"{snr_db:.1f}",
				"" if np.isnan(ber) else f"{ber:.12g}",
				f"{serviceability:.6f}",
				f"{serviceability * 100.0:.3f}",
			])
	if not show_plots:
		plt.close(fig)
	return plot_path, csv_path


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
	ax.set_xlabel("Signal-to-Noise Ratio (SNR) (dB)")
	ax.set_ylabel("Theoretical throughput (Mbps)")
	ax.set_ylim(-0.2, 8.0)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	ax.legend(loc="lower right", fontsize=8.5)
	
	apply_result_diagram_style(fig)
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
	ax.set_xlabel("Signal-to-Noise Ratio (SNR) (dB)")
	ax.set_ylabel("Actual throughput Goodput (Mbps)")
	ax.set_ylim(-0.2, 8.0)
	ax.set_xlim(snr_range_db[0], snr_range_db[-1])
	ax.grid(True, which="both", alpha=0.35)
	ax.legend(loc="lower right", fontsize=8.5)
	
	apply_result_diagram_style(fig)
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
	p_mcs_list: list[MCSConfig | None] = []
	p_ber_theory = []
	p_per_theory = []
	p_throughput = []
	p_goodput = []
	
	# Taxiing stats
	t_mcs_list: list[MCSConfig | None] = []
	t_ber_theory = []
	t_per_theory = []
	t_throughput = []
	t_goodput = []
	
	for x in parking_snr:
		selection = select_mcs(float(x), mcs_table)
		selected = selection.selected_mcs
		p_mcs_list.append(selected)
		if selection.status is MCSStatus.NOT_SERVICEABLE or selected is None:
			p_ber_theory.append(np.nan)
			p_per_theory.append(1.0)
			p_throughput.append(0.0)
			p_goodput.append(0.0)
			continue
		ber = ber_for_mcs(float(x), selected)
		p_ber_theory.append(ber)
		per = 1.0 - (1.0 - ber) ** packet_size_bits
		p_per_theory.append(per)
		tp = throughput_mbps(selected)
		p_throughput.append(tp)
		p_goodput.append(tp * (1.0 - per))
		
	for x in taxiing_snr:
		selection = select_mcs(float(x), mcs_table)
		selected = selection.selected_mcs
		t_mcs_list.append(selected)
		if selection.status is MCSStatus.NOT_SERVICEABLE or selected is None:
			t_ber_theory.append(np.nan)
			t_per_theory.append(1.0)
			t_throughput.append(0.0)
			t_goodput.append(0.0)
			continue
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
	p_mcs_idx = [-1 if mcs is None else mcs_order[mcs.name] for mcs in p_mcs_list]
	t_mcs_idx = [-1 if mcs is None else mcs_order[mcs.name] for mcs in t_mcs_list]
	
	# Row 0: SNR
	axes[0, 0].plot(frame_indices, parking_snr, color="tab:blue", linewidth=1.5, label="SNR")
	axes[0, 0].axhline(parking_scenario.mean_snr_db, color="darkblue", linestyle="--", linewidth=1.0, label="Mean")
	axes[0, 0].set_ylabel("SNR (dB)")
	axes[0, 0].set_title(f"Parking Scenario (NLOS, Mean = 8.0 dB, K = {parking_scenario.k_factor_db:.1f} dB)")
	axes[0, 0].grid(True, alpha=0.3)
	axes[0, 0].legend(loc="upper right", fontsize=8)
	
	axes[0, 1].plot(frame_indices, taxiing_snr, color="tab:blue", linewidth=1.5, label="SNR")
	axes[0, 1].axhline(taxiing_scenario.mean_snr_db, color="darkblue", linestyle="--", linewidth=1.0, label="Mean")
	axes[0, 1].set_ylabel("SNR (dB)")
	axes[0, 1].set_title(f"Taxiing Scenario (LOS, Mean = 15.5 dB, K = {taxiing_scenario.k_factor_db:.1f} dB)")
	axes[0, 1].grid(True, alpha=0.3)
	axes[0, 1].legend(loc="upper right", fontsize=8)
	
	# Row 1: MCS Selection
	axes[1, 0].step(frame_indices, p_mcs_idx, where="mid", color="tab:orange", linewidth=1.5)
	axes[1, 0].set_yticks([-1, *range(len(mcs_names))])
	axes[1, 0].set_yticklabels(["OUTAGE", *mcs_names], fontsize=8)
	axes[1, 0].set_ylim(-1.5, len(mcs_names) - 0.5)
	axes[1, 0].set_ylabel("Selected MCS / status")
	axes[1, 0].set_title(f"Outage: {sum(mcs is None for mcs in p_mcs_list)}/{frames} frames")
	axes[1, 0].grid(True, alpha=0.3)
	
	axes[1, 1].step(frame_indices, t_mcs_idx, where="mid", color="tab:orange", linewidth=1.5)
	axes[1, 1].set_yticks([-1, *range(len(mcs_names))])
	axes[1, 1].set_yticklabels(["OUTAGE", *mcs_names], fontsize=8)
	axes[1, 1].set_ylim(-1.5, len(mcs_names) - 0.5)
	axes[1, 1].set_ylabel("Selected MCS / status")
	axes[1, 1].set_title(f"Outage: {sum(mcs is None for mcs in t_mcs_list)}/{frames} frames")
	axes[1, 1].grid(True, alpha=0.3)
	
	# Row 2: BER & PER
	axes[2, 0].semilogy(frame_indices, p_ber_theory, color="tab:green", linewidth=1.2, label="Theoretical BER")
	axes[2, 0].semilogy(frame_indices, p_per_theory, color="tab:red", linewidth=1.2, label="PER (4096 bits)")
	axes[2, 0].axhline(TARGET_BER, color="black", linestyle="--", linewidth=0.9, label="Target BER = 1e-6")
	axes[2, 0].set_ylabel("Error rate")
	# Enforce PER lower y-limit for consistent comparison (10^-12)
	axes[2, 0].set_ylim(1e-12, 1.0)
	axes[2, 0].grid(True, which="both", alpha=0.3)
	axes[2, 0].legend(loc="upper right", fontsize=8)
	
	axes[2, 1].semilogy(frame_indices, t_ber_theory, color="tab:green", linewidth=1.2, label="Theoretical BER")
	axes[2, 1].semilogy(frame_indices, t_per_theory, color="tab:red", linewidth=1.2, label="PER (4096 bits)")
	axes[2, 1].axhline(TARGET_BER, color="black", linestyle="--", linewidth=0.9, label="Target BER = 1e-6")
	axes[2, 1].set_ylabel("Error rate")
	# Enforce PER lower y-limit for consistent comparison (10^-12)
	axes[2, 1].set_ylim(1e-12, 1.0)
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
	apply_result_diagram_style(fig)
	
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "fig8_amc_timeseries_simulation.png"
	fig.savefig(output_path, dpi=160, bbox_inches="tight")

	# Additionally, produce a per-base-station PER time-series figure for TSN
	parking_snr_bs, parking_ber_bs, parking_per_bs, parking_outage_bs = simulate_per_per_bs(parking_scenario, mcs_table, frames, packet_size_bits, rng)
	taxi_snr_bs, taxi_ber_bs, taxi_per_bs, taxi_outage_bs = simulate_per_per_bs(taxiing_scenario, mcs_table, frames, packet_size_bits, rng)

	fig2, ax2 = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
	bs_count = int(parking_per_bs.shape[0])
	for i in range(bs_count):
		ax2[0].semilogy(frame_indices, parking_per_bs[i, :], linewidth=1.2, label=f"BS {i+1}")
		ax2[1].semilogy(frame_indices, taxi_per_bs[i, :], linewidth=1.2, label=f"BS {i+1}")
		parking_outage_frames = frame_indices[parking_outage_bs[i, :]]
		taxi_outage_frames = frame_indices[taxi_outage_bs[i, :]]
		if parking_outage_frames.size:
			ax2[0].scatter(parking_outage_frames, np.ones(parking_outage_frames.size), marker="x", color="black", s=18, label="Outage" if i == 0 else None, zorder=3)
		if taxi_outage_frames.size:
			ax2[1].scatter(taxi_outage_frames, np.ones(taxi_outage_frames.size), marker="x", color="black", s=18, label="Outage" if i == 0 else None, zorder=3)

	ax2[0].set_title("Parking: PER per base-station (time-series)")
	ax2[0].set_ylabel("PER")
	# Keep lower PER limit at 1e-12 for consistent visualization
	ax2[0].set_ylim(1e-12, 1.0)
	ax2[0].grid(True, which="both", alpha=0.3)
	ax2[0].legend(loc="upper right", fontsize=8, ncol=min(4, bs_count))

	ax2[1].set_title("Taxiing: PER per base-station (time-series)")
	ax2[1].set_ylabel("PER")
	# Keep lower PER limit at 1e-12 for consistent visualization
	ax2[1].set_ylim(1e-12, 1.0)
	ax2[1].set_xlabel("Frame index")
	ax2[1].grid(True, which="both", alpha=0.3)
	ax2[1].legend(loc="upper right", fontsize=8, ncol=min(4, bs_count))

	fig2.tight_layout()
	apply_result_diagram_style(fig2)
	per_out_path = output_dir / "fig8_per_by_bs_timeseries.png"
	fig2.savefig(per_out_path, dpi=160, bbox_inches='tight')

	if not show_plots:
		plt.close(fig)
		plt.close(fig2)
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
	
	apply_result_diagram_style(fig)
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
	apply_result_diagram_style(fig)
	
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
	apply_result_diagram_style(fig)
	
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
	parser.add_argument("--base-station-count", type=int, default=8, help="Number of BS sites in the selected airport layout")
	parser.add_argument("--layout-profile", type=str, default="TSN_8BS", help="Station-layout calibration profile")
	parser.add_argument("--output-dir", type=str, help="Directory to save output plots")
	parser.add_argument("--frequency-ghz", type=float, default=5.12, help="Carrier frequency in GHz; documented project value")
	parser.add_argument("--bandwidth-mhz", type=float, default=5.0, help="Channel bandwidth in MHz; documented project value")
	parser.add_argument("--tx-eirp-dbm", type=float, default=39.4, help="Maximum BS EIRP in dBm; documented upper limit")
	parser.add_argument("--aircraft-gain-dbi", type=float, default=6.0, help="Aircraft antenna gain in dBi")
	parser.add_argument("--noise-figure-db", type=float, default=8.0, help="Aircraft receiver noise figure in dB")
	parser.add_argument("--implementation-loss-db", type=float, default=5.0, help="Implementation/deployment loss in dB")
	parser.add_argument("--los-path-exponent", type=float, default=3.0, help="LoS scenario path-loss exponent")
	parser.add_argument("--nlos-path-exponent", type=float, default=4.0, help="NLoS scenario path-loss exponent")
	parser.add_argument("--los-obstacle-loss-db", type=float, default=4.0, help="LoS additional obstacle loss in dB")
	parser.add_argument("--nlos-obstacle-loss-db", type=float, default=15.0, help="NLoS additional obstacle loss in dB")
	parser.add_argument("--downlink-eirp-dbm", type=float, default=37.4, help="Equivalent BS downlink EIRP from ICAO DOC 10044 table")
	parser.add_argument("--uplink-eirp-dbm", type=float, default=24.0, help="Aircraft uplink EIRP from supplied ICAO table")
	parser.add_argument("--receiver-sensitivity-dbm", type=float, default=-89.37, help="Target receiver sensitivity in dBm")
	parser.add_argument("--fade-margin-db", type=float, default=14.0, help="Total fade margin from supplied ICAO link-budget table")
	parser.add_argument("--downlink-mapl-db", type=float, default=116.77, help="Maximum allowable downlink path loss in dB")
	parser.add_argument("--uplink-mapl-db", type=float, default=115.37, help="Maximum allowable uplink path loss in dB")
	parser.add_argument(
		"--k-values-db",
		type=lambda value: [float(item.strip()) for item in value.split(",") if item.strip()],
		default=[-10.0, -5.0, 0.0, 5.0, 10.0, 15.0],
		help="Comma-separated Rician K sensitivity scenarios in dB (not TSN measurements)",
	)
	parser.add_argument("--k-samples", type=int, default=100000, help="Monte Carlo samples per Rician K value")
	parser.add_argument("--k-seed", type=int, default=20261120, help="Random seed for reproducible K sensitivity")
	parser.add_argument(
		"--k-sensitivity-only",
		action="store_true",
		help="Run only the TSN point-link Rician K sensitivity sweep",
	)
	return parser.parse_args()


def print_table(mcs_table: list[MCSConfig]) -> None:
	print("MCS table")
	print("name\t\tmod\tcode\trep\tthreshold_dB\tthroughput_Mbps")
	for mcs in mcs_table:
		print(
			f"{mcs.name}\t{mcs.modulation_order}\t{mcs.code_rate:.2f}\t{mcs.repetition}\t"
			f"{mcs.snr_threshold_db:>7.1f}\t\t{throughput_mbps(mcs):.3f}"
		)


def calculate_coverage_radius_m(
	snr_threshold_db: float,
	config: TSNLinkBudgetConfig,
	distance_min_m: float = 1.0,
	distance_max_m: float = 50000.0,
) -> float:
	"""Maximum 2-D great-circle distance for a median SNR threshold."""
	low = distance_min_m
	high = distance_max_m
	for _ in range(60):
		mid = (low + high) / 2.0
		snr_db = calculate_tsn_link_budget(mid, config)["snr_db"]
		if snr_db >= snr_threshold_db:
			low = mid
		else:
			high = mid
	return (low + high) / 2.0


def print_tsn_test_point_link_budget(args: argparse.Namespace, output_dir: Path) -> None:
	layout = build_tsn_bs_layout()
	test_points = build_tsn_test_points()
	frequency_hz = args.frequency_ghz * 1e9
	bandwidth_hz = args.bandwidth_mhz * 1e6
	output_dir.mkdir(parents=True, exist_ok=True)

	print("\n3-D link budget at P1-P5 (17 m AGL)")
	print(
		"point\tnearest_bs\thorizontal_m\tdelta_height_m\tslant_m\tcondition\t"
		"path_loss_db\tdownlink_rx_dbm\tdownlink_margin_db\tuplink_rx_dbm\tuplink_margin_db"
	)
	fields = [
		"point", "nearest_bs", "horizontal_m", "delta_height_m", "slant_m", "condition",
		"path_loss_db", "downlink_rx_dbm", "downlink_margin_db",
		"uplink_rx_dbm", "uplink_margin_db",
	]
	with (output_dir / "tsn_test_points_link_budget.csv").open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fields)
		writer.writeheader()
		for point in test_points:
			distances = [
				(
					haversine_distance(point.latitude, point.longitude, station.latitude, station.longitude),
					station,
				)
				for station in layout
			]
			horizontal_m, station = min(distances, key=lambda item: item[0])
			delta_height_m = station.height_agl_m - point.height_agl_m
			slant_m = sqrt(horizontal_m ** 2 + delta_height_m ** 2)
			path_exponent = args.los_path_exponent if station.line_of_sight else args.nlos_path_exponent
			obstacle_loss_db = args.los_obstacle_loss_db if station.line_of_sight else args.nlos_obstacle_loss_db
			path_loss = calculate_tsn_path_loss(
				distance_m=slant_m,
				frequency_hz=frequency_hz,
				path_loss_exponent=path_exponent,
				reference_distance_m=1.0,
				obstacle_loss_db=obstacle_loss_db,
			)
			downlink_rx_dbm = args.downlink_eirp_dbm - path_loss["path_loss_db"]
			uplink_rx_dbm = args.uplink_eirp_dbm - path_loss["path_loss_db"]
			downlink_margin_db = args.downlink_mapl_db - path_loss["path_loss_db"]
			uplink_margin_db = args.uplink_mapl_db - path_loss["path_loss_db"]
			condition = "LoS" if station.line_of_sight else "NLoS"
			writer.writerow({
				"point": point.point_id,
				"nearest_bs": station.station_id,
				"horizontal_m": f"{horizontal_m:.1f}",
				"delta_height_m": f"{delta_height_m:.1f}",
				"slant_m": f"{slant_m:.1f}",
				"condition": condition,
				"path_loss_db": f"{path_loss['path_loss_db']:.2f}",
				"downlink_rx_dbm": f"{downlink_rx_dbm:.2f}",
				"downlink_margin_db": f"{downlink_margin_db:.2f}",
				"uplink_rx_dbm": f"{uplink_rx_dbm:.2f}",
				"uplink_margin_db": f"{uplink_margin_db:.2f}",
			})
			print(
				f"{point.point_id}\t{station.station_id}\t{horizontal_m:.1f}\t"
				f"{delta_height_m:.1f}\t{slant_m:.1f}\t{condition}\t"
				f"{path_loss['path_loss_db']:.2f}\t{downlink_rx_dbm:.2f}\t"
				f"{downlink_margin_db:.2f}\t{uplink_rx_dbm:.2f}\t{uplink_margin_db:.2f}"
			)


def run_tsn_k_sensitivity(
	args: argparse.Namespace,
	output_dir: Path,
	k_values_db: list[float],
) -> Path:
	if args.k_samples < 1:
		raise ValueError("--k-samples must be >= 1")
	if not k_values_db:
		raise ValueError("--k-values-db must contain at least one K value")

	layout = build_tsn_bs_layout()
	test_points = build_tsn_test_points()
	frequency_hz = args.frequency_ghz * 1e9
	rng = np.random.default_rng(args.k_seed)
	scatter = (
		rng.normal(0.0, 1.0, args.k_samples)
		+ 1j * rng.normal(0.0, 1.0, args.k_samples)
	) / sqrt(2.0)
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "tsn_k_sensitivity.csv"
	fields = [
		"point", "best_candidate_bs", "condition", "horizontal_distance_m", "slant_distance_m",
		"k_db", "direction", "path_loss_db", "mean_link_margin_db",
		"rician_mean_power_linear", "fading_p05_db", "fading_median_db",
		"availability_percent", "outage_percent", "samples", "seed",
	]

	print("\nRician K sensitivity (scenario sweep; not field-calibrated)")
	print("point\tbest_candidate_bs\tcondition\tK_dB\tdirection\tmean_margin_dB\tavailability_%\toutage_%")
	with output_path.open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fields)
		writer.writeheader()
		for point in test_points:
			candidates = []
			for bs in layout:
				horizontal_distance_m = haversine_distance(
					point.latitude, point.longitude, bs.latitude, bs.longitude
				)
				slant_distance_m = sqrt(
					horizontal_distance_m ** 2 + (bs.height_agl_m - point.height_agl_m) ** 2
				)
				path_exponent = args.los_path_exponent if bs.line_of_sight else args.nlos_path_exponent
				obstacle_loss_db = args.los_obstacle_loss_db if bs.line_of_sight else args.nlos_obstacle_loss_db
				path_loss_db = calculate_tsn_path_loss(
					distance_m=slant_distance_m,
					frequency_hz=frequency_hz,
					path_loss_exponent=path_exponent,
					reference_distance_m=1.0,
					obstacle_loss_db=obstacle_loss_db,
				)["path_loss_db"]
				candidates.append((path_loss_db, horizontal_distance_m, slant_distance_m, bs))
			path_loss_db, horizontal_distance_m, slant_distance_m, station = min(
				candidates, key=lambda candidate: candidate[0]
			)
			condition = "LoS" if station.line_of_sight else "NLoS"
			link_gains = {
				"downlink": args.downlink_mapl_db + args.fade_margin_db,
				"uplink": args.uplink_mapl_db + args.fade_margin_db,
			}

			for k_db in k_values_db:
				if not np.isfinite(k_db):
					raise ValueError("K values must be finite dB numbers")
				k_linear = 10.0 ** (k_db / 10.0)
				h = sqrt(k_linear / (k_linear + 1.0)) + scatter / sqrt(k_linear + 1.0)
				fading_db = 10.0 * np.log10(np.maximum(np.abs(h) ** 2, np.finfo(float).tiny))
				for direction, link_gain_db in link_gains.items():
					mean_margin_db = link_gain_db - path_loss_db
					available = mean_margin_db + fading_db >= 0.0
					availability_percent = float(np.mean(available) * 100.0)
					outage_percent = 100.0 - availability_percent
					writer.writerow({
						"point": point.point_id,
						"best_candidate_bs": station.station_id,
						"condition": condition,
						"horizontal_distance_m": f"{horizontal_distance_m:.2f}",
						"slant_distance_m": f"{slant_distance_m:.2f}",
						"k_db": f"{k_db:.2f}",
						"direction": direction,
						"path_loss_db": f"{path_loss_db:.2f}",
						"mean_link_margin_db": f"{mean_margin_db:.2f}",
						"rician_mean_power_linear": "1.0",
						"fading_p05_db": f"{np.percentile(fading_db, 5):.3f}",
						"fading_median_db": f"{np.median(fading_db):.3f}",
						"availability_percent": f"{availability_percent:.3f}",
						"outage_percent": f"{outage_percent:.3f}",
						"samples": args.k_samples,
						"seed": args.k_seed,
					})
					print(
						f"{point.point_id}\t{station.station_id}\t{condition}\t{k_db:.1f}\t"
						f"{direction}\t{mean_margin_db:.2f}\t"
						f"{availability_percent:.3f}\t{outage_percent:.3f}"
					)
	return output_path


def print_tsn_geographic_link_model(
	args: argparse.Namespace,
	mcs_table: list[MCSConfig],
	output_dir: Path,
) -> None:
	layout = build_tsn_bs_layout()
	frequency_hz = args.frequency_ghz * 1e9
	bandwidth_hz = args.bandwidth_mhz * 1e6
	output_dir.mkdir(parents=True, exist_ok=True)

	print("\nTSN WGS84 base-station layout")
	with (output_dir / "tsn_bs_layout.csv").open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=["station_id", "name", "latitude", "longitude", "condition"])
		writer.writeheader()
		for station in layout:
			writer.writerow({
				"station_id": station.station_id,
				"name": station.name,
				"latitude": f"{station.latitude:.5f}",
				"longitude": f"{station.longitude:.5f}",
				"condition": "LoS" if station.line_of_sight else "NLoS",
			})
			print(
				f"{station.station_id}\t{station.name}\t"
				f"lat={station.latitude:.5f}\tlon={station.longitude:.5f}\t"
				f"{'LoS' if station.line_of_sight else 'NLoS'}"
			)

	print("\nInter-BS great-circle distances and link budgets")
	print("pair\tdistance_m\tcondition\tpath_loss_db\trx_power_dbm\tnoise_floor_dbm\tsnr_db")
	pair_fields = [
		"station_a", "station_b", "distance_m", "condition",
		"path_loss_db", "received_power_dbm", "noise_floor_dbm", "snr_db",
	]
	with (output_dir / "tsn_inter_bs_link_budget.csv").open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=pair_fields)
		writer.writeheader()
		for i, station_a in enumerate(layout):
			for station_b in layout[i + 1 :]:
				distance_m = haversine_distance(
					station_a.latitude,
					station_a.longitude,
					station_b.latitude,
					station_b.longitude,
				)
				line_of_sight = station_a.line_of_sight and station_b.line_of_sight
				path_exponent = args.los_path_exponent if line_of_sight else args.nlos_path_exponent
				obstacle_loss_db = args.los_obstacle_loss_db if line_of_sight else args.nlos_obstacle_loss_db
				config = TSNLinkBudgetConfig(
					frequency_hz=frequency_hz,
					bandwidth_hz=bandwidth_hz,
					tx_eirp_dbm=args.tx_eirp_dbm,
					aircraft_antenna_gain_dbi=args.aircraft_gain_dbi,
					receiver_noise_figure_db=args.noise_figure_db,
					implementation_loss_db=args.implementation_loss_db,
					path_loss_exponent=path_exponent,
					reference_distance_m=1.0,
					obstacle_loss_db=obstacle_loss_db,
				)
				result = calculate_tsn_link_budget(distance_m, config)
				writer.writerow({
					"station_a": station_a.station_id,
					"station_b": station_b.station_id,
					"distance_m": f"{distance_m:.1f}",
					"condition": "LoS" if line_of_sight else "NLoS",
					"path_loss_db": f"{result['path_loss_db']:.2f}",
					"received_power_dbm": f"{result['received_power_dbm']:.2f}",
					"noise_floor_dbm": f"{result['noise_floor_dbm']:.2f}",
					"snr_db": f"{result['snr_db']:.2f}",
				})
				print(
					f"{station_a.station_id}-{station_b.station_id}\t"
					f"{distance_m:.1f}\t{'LoS' if line_of_sight else 'NLoS'}\t"
					f"{result['path_loss_db']:.2f}\t{result['received_power_dbm']:.2f}\t"
					f"{result['noise_floor_dbm']:.2f}\t{result['snr_db']:.2f}"
				)

	print("\nMedian 2-D coverage radius by MCS threshold (ground-plane distance; height absent)")
	print("station_id\tcondition\tmcs\tsnr_threshold_db\tcoverage_radius_m")
	coverage_fields = ["station_id", "condition", "mcs", "snr_threshold_db", "coverage_radius_m"]
	with (output_dir / "tsn_mcs_coverage_radius.csv").open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=coverage_fields)
		writer.writeheader()
		for station in layout:
			path_exponent = args.los_path_exponent if station.line_of_sight else args.nlos_path_exponent
			obstacle_loss_db = args.los_obstacle_loss_db if station.line_of_sight else args.nlos_obstacle_loss_db
			config = TSNLinkBudgetConfig(
				frequency_hz=frequency_hz,
				bandwidth_hz=bandwidth_hz,
				tx_eirp_dbm=args.tx_eirp_dbm,
				aircraft_antenna_gain_dbi=args.aircraft_gain_dbi,
				receiver_noise_figure_db=args.noise_figure_db,
				implementation_loss_db=args.implementation_loss_db,
				path_loss_exponent=path_exponent,
				reference_distance_m=1.0,
				obstacle_loss_db=obstacle_loss_db,
			)
			for mcs in mcs_table:
				radius_m = calculate_coverage_radius_m(mcs.snr_threshold_db, config)
				writer.writerow({
					"station_id": station.station_id,
					"condition": "LoS" if station.line_of_sight else "NLoS",
					"mcs": mcs.name,
					"snr_threshold_db": f"{mcs.snr_threshold_db:.1f}",
					"coverage_radius_m": f"{radius_m:.1f}",
				})
				print(
					f"{station.station_id}\t{'LoS' if station.line_of_sight else 'NLoS'}\t"
					f"{mcs.name}\t{mcs.snr_threshold_db:.1f}\t{radius_m:.1f}"
				)


def main() -> None:
	args = parse_args()
	mcs_table = build_mcs_table()
	if args.output_dir:
		output_dir = Path(args.output_dir)
	else:
		output_dir = Path(__file__).resolve().parent.parent / "simulation_results"
	if args.k_sensitivity_only:
		output_path = run_tsn_k_sensitivity(args, output_dir, args.k_values_db)
		print(f"\nK sensitivity results saved: {output_path}")
		return

	scenarios = build_scenarios(
		k_parking=args.k_parking,
		k_taxiing=args.k_taxiing,
		base_station_count=args.base_station_count,
		bs_layout_name=args.layout_profile,
	)
	show_plots = not args.headless and os.environ.get("AMC_HEADLESS") != "1"
	if not show_plots:
		plt.switch_backend("Agg")

	print_tsn_geographic_link_model(args, mcs_table, output_dir)
	print_tsn_test_point_link_budget(args, output_dir)
	k_sensitivity_path = run_tsn_k_sensitivity(args, output_dir, args.k_values_db)
	print(f"K sensitivity results saved: {k_sensitivity_path}")
	print_table(mcs_table)
	selection_plot, selection_csv = plot_mcs_serviceability_thresholds(mcs_table, output_dir, show_plots)
	print(f"MCS serviceability plot saved: {selection_plot}")
	print(f"MCS threshold probe data saved: {selection_csv}")
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
	taxiing_status_plot, taxiing_status_csv = plot_taxiing_amc_serviceability(
		scenarios["taxiing"], mcs_table, output_dir, show_plots
	)
	print(f"Saved Taxiing AMC serviceability plot: {taxiing_status_plot}")
	print(f"Saved Taxiing AMC serviceability data: {taxiing_status_csv}")

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