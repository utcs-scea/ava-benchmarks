#include "aos_host_common.h"

class aos_scheduler {
public:

    aos_scheduler();
    void parseImages(std::string fileName);
    void clearImage();
    void loadImage(uint32_t image_idx);
    bool agfiExists(std::string agfi);
    bool anyImageLoaded();
    json getImageByAgfi(std::string agfi);

    bool canImageSatisfyNeed(json & image, json & app_ids);
    json convertImageToAppTuples(json & image);

    int32_t getFittingImageIdx(json & app_ids);

private:

    json image_library;
    json current_image;

};
