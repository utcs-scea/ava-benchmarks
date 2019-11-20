#include "aos_scheduler.h"

aos_scheduler::aos_scheduler() {

}

void aos_scheduler::parseImages(std::string fileName) {

    std::ifstream json_in_file;
    json_in_file.open(fileName);

    json parsed_file = json::parse(json_in_file);

    json_in_file.close();

    if (image_library.empty()) {
        cout << "Scheduler: Image Library is intially empty" << endl << flush;
    }

    cout << "Scheduler: Parsed file " << fileName << " and found " << parsed_file["images"].size() << " images." << endl << flush;

    for (auto & image : parsed_file["images"]) {
        std::string agfi_ = image["agfi"];
        if (agfiExists(agfi_)) {
            cout << "Scheduler: Skipping already exists agfi: " << agfi_ << endl << flush;
        } else {
            image_library.push_back(image);
            cout << "Scheduler: Image Library Adding agfi: " << agfi_ << " Description: " << image["description"] << endl << flush;
        }
    } // for in

    cout << "Scheduler: Image library now has " << image_library.size() << " images." << endl << flush;
}

void aos_scheduler::clearImage() {
	cout << "Sheduler: Preparing to clear FPGA Image" << endl << flush;
    std::string clear_command = "sudo fpga-clear-local-image  -S 0";
    std::string clear_result = cmd_exec(clear_command);
    cout << "Scheduler: Clear result: " << clear_result << endl << flush;
}

void aos_scheduler::loadImage(uint32_t image_idx) {

    if (image_idx > image_library.size()) {
    	cout << "Scheduler: Invalid image selection index." << endl << flush;
    	return;
    }

    if (anyImageLoaded()) {
        cout << "Scheduler: Overwritting current image agfi: " << current_image["agfi"] << " Description: " << current_image["description"] << endl << flush;
    } else {
        cout << "Scheduler: No prior image written" << endl << flush;
    }

	std::string afgi = image_library[image_idx]["agfi"];
	std::string image_desc = image_library[image_idx]["description"];
    std::string load_cmd  = "sudo fpga-load-local-image -S 0 -I a " + afgi;

    cout << "Scheduler: Attempting to load image with index " << image_idx << " afgi: " << afgi << " Description: " << image_desc << endl << flush;

    std::string load_result = cmd_exec(load_cmd);

    cout << "Scheduler: Load result: " << load_result << endl << flush;

    current_image = image_library[image_idx];

}

bool aos_scheduler::agfiExists(std::string agfi) {

    for (auto & image : image_library) {
        std::string agfi_ = image["agfi"];
        if (agfi == agfi_) {
            return true;
        }
    }

    return false;
}

bool aos_scheduler::anyImageLoaded() {
    if (current_image.size() == 0) {
        return false;
    } else {
        return true;
    }
}


json aos_scheduler::getImageByAgfi(std::string agfi) {
    for (auto & image : image_library) {
        std::string agfi_ = image["agfi"];
        if (agfi == agfi_) {
            return image;
        }
    }

    return json();
}

bool aos_scheduler::canImageSatisfyNeed(json & image, json & app_tuples) {

    json image_as_tuple = convertImageToAppTuples(image);

    for (auto & app_id : app_tuples.items()) {
        // Check if the image has the type of app we want
        if (image_as_tuple.find(app_id.key()) == image_as_tuple.end()) {
            return false;
        }
        // Now check if it has enough of them
        if (image_as_tuple[app_id.key()]["count"] < app_id.value()["count"]) {
            return false;
        }
    } 

    // All checks have passed
    return true;
}

int32_t aos_scheduler::getFittingImageIdx(json & app_ids) {

    int index = 0;
    for (auto & image : image_library) {
        if (canImageSatisfyNeed(image , app_ids)) {
            return index;
        }
        index++;
    }

    return -1;
}

json aos_scheduler::convertImageToAppTuples(json & image) {
    json app_tuples;
    for (auto & slot : image["slots"]) {
        std::string app_id = slot["app_id"];
        if (app_tuples.find(app_id) != app_tuples.end()) {
            // update count of existing app_id
            uint32_t current_count = app_tuples[app_id]["count"];
            app_tuples[app_id]["count"] = (current_count + 1);
        } else {
            // adding new entry
            app_tuples[app_id]["count"]  = 1;
            //app_tuples[app_id]["app_id"] = std::string(); // replicated
        }
    }
    return app_tuples;
}



