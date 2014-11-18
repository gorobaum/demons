#ifndef DEMONS_DEMON_H_
#define DEMONS_DEMON_H_

class Demon {
public:
	Demon(std::vector<int> position, std::vector<int> volumeOfInfluence) :
		position_(position),
		volumeOfInfluence_(volumeOfInfluence) {}
	std::vector<int> getPosition() { return position_; }
	std::vector<int> getVolumeOfInfluence() { return volumeOfInfluence_; }
private:
	std::vector<int> position_;
	std::vector<int> volumeOfInfluence_;
};

#endif