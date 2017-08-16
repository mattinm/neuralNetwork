#include <string>
#include <map>
#include <unordered_map>
#include <vector>

struct Box
{
	int32_t species_id;
	int32_t x;  // top left corner x
	int32_t y;  // top left corner y
	int32_t w;  // width
	int32_t h;  // height
	int32_t cx; // center point x
	int32_t cy; // center point y
	int32_t ex; // bottom right corner x
	int32_t ey; // bottom right corner y
	uint64_t user_high, user_low;
	bool matched = false; // whether we found a matching observation
	bool hit = false; //whether we looked at this msi at all


	Box();
	Box(int32_t species_id, int32_t x, int32_t y, int32_t w, int32_t h, uint64_t user_high, uint64_t user_low);
	void load(int32_t species_id, int32_t x, int32_t y, int32_t w, int32_t h, uint64_t user_high, uint64_t user_low);
	std::string toString();
};

struct MSI
{
	int msi;
	bool used = false;
	std::vector<Box> boxes;

	//bmr => background_misclassified_percentages<species_id, ratio BG classified as species_id>
	//for the species_id != BACKGROUND, it is the misclassified ratio
	//for the species_id == BACKGROUND, it is the correctly classified ratio
	// unordered_map<int,float> bmr;
	std::unordered_map<int,int> bmc; // <species_id, pixel count of BG classified as species_id>
	int totalBGCount = 0;
	int numPixels;
	std::string original_image_path = "";

	MSI();
	MSI(int msi, int numBoxes);
	void init(int msi, int numBoxes);
};

void readLocationsFile(const char* filename, std::map<int,MSI> &locations);

int getMSI(std::string filename);