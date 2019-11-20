#include "aos_app_session.h"

aos_app_session::aos_app_session(std::string app_id, session_id_t session_id): 
    app_id(app_id),
    session_id(session_id),
    active_slot(false),
    fpga_slot(~0x0),
    saved_state(false)
{

}

void aos_app_session::unbindFromSlot() {
    active_slot = false;
    fpga_slot   = (~0x0);
}

void aos_app_session::bindToSlot(uint64_t slot_id) {
    active_slot = true;
    fpga_slot   = slot_id;
}

bool aos_app_session::boundToSlot() {
    return active_slot;
}

uint64_t aos_app_session::getSlotId() {
    return fpga_slot;
}

bool aos_app_session::hasSavedState() {
    return saved_state;
}

std::string aos_app_session::debugString() const {
    std::string toRet = "SID: ";
    toRet += session_id;
    toRet += " AppId: ";
    toRet += app_id;
    toRet += " Scheduled: ";
    toRet += (active_slot ? " Yes" : " No");
    toRet += " Slot: ";
    if (active_slot) {
        toRet += fpga_slot;
    } else {
        toRet += "NONE";
    }
    toRet += (saved_state ? " Yes" : " No");
    return toRet;
}

session_id_t aos_app_session::getSessionId() const {
    return session_id;
}