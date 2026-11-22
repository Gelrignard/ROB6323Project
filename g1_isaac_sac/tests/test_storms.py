"""Keep official wind-speed magnitudes and engineering transformations reproducible."""
import math
import pytest
import torch
from g1_wind_sac_v5.storms import STORM_TARGETS_MPS, REFERENCE_CLASSES, storm_curriculum_ceiling, storm_min_wind
from g1_wind_sac_v5.aerodynamics import power_law_wind_profile


def test_reference_ranges_and_open_ended_cat5():
    assert REFERENCE_CLASSES['hurricane_cat1'].lower_mps == pytest.approx(74 * .44704)
    assert REFERENCE_CLASSES['hurricane_cat1'].upper_mps == pytest.approx(95 * .44704)
    assert REFERENCE_CLASSES['hurricane_cat5'].upper_mps is None
    assert REFERENCE_CLASSES['ef2_gust'].averaging == 'estimated 3-second gust'
    assert STORM_TARGETS_MPS['hurricane_cat4'] == 70


def test_staged_schedule_has_increasing_ceiling_and_holds_final():
    x = [storm_curriculum_ceiling(s,2,43,10000) for s in range(0,12001,200)]
    assert x[0] == 2
    assert x[-1] == 43
    assert all(a <= b for a,b in zip(x, x[1:]))
    assert storm_curriculum_ceiling(8000,2,43,10000) < 43
    assert storm_curriculum_ceiling(14000,2,43,10000) == 43


def test_storm_min_fraction():
    assert storm_min_wind(43,.3)==pytest.approx(12.9)
    with pytest.raises(ValueError): storm_min_wind(10,1.2)


def test_shear_profile_keeps_reference_and_reduces_near_ground():
    u = torch.tensor([[[43.,0.,0.],[43.,0.,0.],[43.,0.,0.]]])
    z = torch.tensor([[.5,1.0,10.0]])
    v = power_law_wind_profile(u,z,ref_height_m=10.,exponent=.14)
    assert torch.allclose(v[0,2],u[0,2])
    assert v[0,0,0] < v[0,1,0] < v[0,2,0]
    assert v[0,1,0].item() == pytest.approx(43*(1/10)**.14, rel=1e-6)
    assert torch.allclose(power_law_wind_profile(u,z,exponent=0),u)
