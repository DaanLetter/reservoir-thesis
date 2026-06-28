import logging

import numpy as np
import matplotlib.pyplot as plt

#based on Steyaert2025: Data derived reservoir operations simulated in a global hydrologic model

logger = logging.getLogger(__name__)

BANKFULL_NUMBER = 2.3 #ratio of bankfull discharge to the average discharge

def reduction_factor(current_storage, min_storage, max_storage, storage_capacity=None):
    if current_storage <= max_storage:
        # Normal zone: conservation → flood curve
        rf = (current_storage - min_storage) / (max_storage - min_storage)
    else:
        # Surplus zone: flood curve → capacity
        if storage_capacity is not None and storage_capacity > max_storage:
            rf = (current_storage - max_storage) / (storage_capacity - max_storage) #jen wrote flood curve - capacity  but i believe that taht would give a negative rf
        else:
            rf = 1.0
    return max(0.0, min(1.0, rf))

def demand_reduction_factor(current_storage, storage_capacity, max_storage):
    # max_storage is the flood bound here
    denom = max_storage - 0.1 * storage_capacity
    if denom <= 0:
        return 1.0
    rf = (current_storage - 0.1 * storage_capacity) / denom
    return max(0.0, min(1.0, rf))

def initial_discharge(current_storage, max_storage, min_storage, avg_outflow, storage_capacity):

    #equation 1

    RF = reduction_factor(current_storage, min_storage, max_storage, storage_capacity)
    Ri = avg_outflow*RF
    return Ri

def new_storage(release, current_storage, inflow=0, precipitation=0, evaporation=0, storage_capacity=None):

    #equation 2
    # FIX: added optional storage_capacity cap  any volume above physical capacity spills immediately
    Sn = current_storage + inflow + precipitation - evaporation - release

    if storage_capacity is not None:
        Sn = min(Sn, storage_capacity)
    Sn = max(Sn, 0) #make sure modeled storage cant be negative. 
    return Sn

def flood_release(current_storage, release, max_storage):

    #determines the extra flood release when required

    return max(current_storage-release-max_storage, 0)

def generic_release(current_storage, avg_discharge, storage_capacity, bankfull_discharge):

    #equation 4, performing the generic reservoir scheme in PCR-GLOBWB 2. Smin and Smax are set at 10 and 75 percent of storage capacity as the active zone of a resevoir.
    #in all instances, demand is set to 0.

    min_storage = 0.1*storage_capacity
    max_storage = 0.75*storage_capacity
    if current_storage < min_storage:
        release = 0
    elif min_storage < current_storage < max_storage:
        release = reduction_factor(current_storage, min_storage, max_storage)*avg_discharge
    elif current_storage > max_storage:
        release = avg_discharge + (current_storage - max_storage) / (storage_capacity - max_storage) * (bankfull_discharge - avg_discharge)

    if current_storage-release > max_storage: #check for flood conditions.
        release = release + flood_release(current_storage, release, max_storage)
    return release


def starfit_release(current_storage, storage_capacity, max_storage, min_storage, avg_outflow, env_flow, demand, current_release=0, use='Hydroelectricity'):

    #equation 6.
    # Irrigation and Water Supply dams use conservation floor of 10% capacity;
    # all other use types (including Hydroelectricity) keep the STARFIT conservation level.
    irrigation_types = {'Irrigation', 'Water Supply'}
    if use in irrigation_types:
        min_storage = 0.1 * storage_capacity

    release = 0

    ID = initial_discharge(current_storage, max_storage, min_storage, avg_outflow, storage_capacity)
    Ri = irrigation_release(min_storage, max_storage, current_storage, current_release, demand, storage_capacity)
    Rh = Hydroelectricity_release(current_storage, min_storage, max_storage, current_release, demand, storage_capacity)

    # Step 1: determine release from storage zone
    if current_storage >= max_storage:
        # Surplus zone: bring storage back to flood bound plus base discharge
        release = (current_storage - max_storage) + ID
    elif min_storage < current_storage < max_storage and use in irrigation_types:
        release = Ri
    elif min_storage < current_storage < max_storage:
        release = Rh
    # below min_storage: release stays 0 (env flow applied below)

    # Step 2: enforce environmental flow floor (Jen point 3, PCR-GLOBWB line 969)
    # If planned release is below env_flow but there is enough water, release at least env_flow.
    if release < env_flow and current_storage - env_flow > 0:
        release = env_flow

    # Step 3: flood release at end (Jen point 1, PCR-GLOBWB lines 982-983)
    # After all operational releases, spill any volume still above physical capacity.
    flood_diff = current_storage - release - storage_capacity
    if flood_diff > 0:
        release += flood_diff

    return release

def initial_hydro_release(current_release, demand, current_storage, max_storage, storage_capacity):

    #equation 7
    # Uses demand_reduction_factor (scaled from 10% capacity to flood bound) per Jen's
    # feedback and PCR-GLOBWB waterBodies.py line 885.

    RF = demand_reduction_factor(current_storage, storage_capacity, max_storage)
    release = demand * RF
    return release

def Hydroelectricity_release(current_storage, min_storage, max_storage, current_release, demand, storage_capacity):

    #equation 8
    # FIX: was returning current_storage - Rhi, which drained the reservoir in one step.
    # Correct behaviour: release Rhi (the scaled demand), capped so storage stays above min.

    release = 0

    Rhi = initial_hydro_release(current_release, demand, current_storage, max_storage, storage_capacity)

    if min_storage < current_storage < max_storage and current_storage - Rhi > min_storage:
        release = Rhi
    elif current_storage - Rhi < min_storage:
        release = max(current_storage-min_storage, 0)
    return release

def irrigation_release(min_storage, max_storage, current_storage, current_release, demand, storage_capacity):

    #equation 9

    release = 0

    RF = demand_reduction_factor(current_storage, storage_capacity, max_storage)

    if min_storage < current_storage < max_storage and current_release > demand:
        release = current_release
    elif min_storage < current_storage < max_storage and current_release < demand:
        release = RF*demand
    if current_storage - release < 0.1*storage_capacity:
        release = max(current_storage-0.1*storage_capacity, 0)
    return release


def main():
    storagelist = []
    timesteps = np.arange(0, 100, 1)
    min_storage, max_storage, current_storage, avg_outflow = 10, 75, 50, 1
    storage_capacity, bankfull_discharge = 100, 2.3

    for t in timesteps:
        release = generic_release(current_storage, avg_outflow, storage_capacity, bankfull_discharge)
        current_storage = current_storage - release  # simplified water balance
        storagelist.append(current_storage)

    plt.plot(timesteps, storagelist)
    plt.show()

if __name__ == '__main__':
    main()
