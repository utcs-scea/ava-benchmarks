#include "aos_host_common.h"

class aos_host;

class aos_app_session {
public:

    friend class ::aos_host;
    aos_app_session(std::string app_id, session_id_t session_id);
    void unbindFromSlot();
    void bindToSlot(uint64_t slot_id);
    bool boundToSlot();
    uint64_t getSlotId();
    bool hasSavedState();
    std::string debugString() const;
    session_id_t getSessionId() const;

private:

    std::string app_id;
    session_id_t session_id;
    bool active_slot;    
    uint64_t fpga_slot;
    bool saved_state;

};