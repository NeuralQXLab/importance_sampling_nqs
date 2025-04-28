# For recording acceptance rate
def acceptance_callback(step, log_data, driver) -> bool:
    log_data["Acceptance"] = driver.state.sampler_state.acceptance
    return True
