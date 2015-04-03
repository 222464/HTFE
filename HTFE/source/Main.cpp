#include <htfe/HTFE.h>

#include <time.h>
#include <iostream>

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram program;

	program.loadFromFile("resources/htfe.cl", cs);

	htfe::HTFE test;

	std::vector<htfe::LayerDesc> layerDescs(3);

	layerDescs[0]._spatialWidth = 32;
	layerDescs[0]._spatialHeight = 32;
	layerDescs[0]._temporalWidth = 64;
	layerDescs[0]._temporalHeight = 64;
	layerDescs[0]._reconstructionRadius = 8;

	layerDescs[1]._spatialWidth = 22;
	layerDescs[1]._spatialHeight = 22;
	layerDescs[1]._temporalWidth = 44;
	layerDescs[1]._temporalHeight = 44;

	layerDescs[2]._spatialWidth = 16;
	layerDescs[2]._spatialHeight = 16;
	layerDescs[2]._temporalWidth = 32;
	layerDescs[2]._temporalHeight = 32;

	test.createRandom(cs, program, 4, 4, layerDescs, -0.1f, 0.1f);

	float sequence[16][16] = {
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }
	};

	for (int t = 0; t < 64; t++) {
		for (int s = 0; s < 16; s++) {
			for (int e = 0; e < 16; e++)
				test.setInput(e, sequence[s][e]);

			test.activate(cs, generator);
			test.learn(cs);
			test.stepEnd();
		}

		//test.clearMemory(cs);
	}

	std::vector<float> prediction(16, 0.0f);

	int errorCount = 0;
	int totalCount = 0;

	for (int s = 0; s < 16; s++) {
		for (int e = 0; e < 16; e++)
			test.setInput(e, sequence[s][e]);

		if (s > 0) {
			bool hasError = false;

			for (int e = 0; e < 16; e++)
				if ((sequence[s][e] > 0.5f) != (prediction[e] > 0.5f))
					hasError = true;

			if (hasError)
				errorCount++;

			totalCount++;
		}

		test.activate(cs, generator);
		test.stepEnd();

		for (int e = 0; e < 16; e++) {
			std::cout << (test.getPrediction(e) > 0.5f ? "1" : "0") << " ";
			prediction[e] = test.getPrediction(e);
		}

		std::cout << std::endl;
	}

	std::cout << "Error: " << (static_cast<float>(errorCount) / totalCount * 100.0f) << "%" << std::endl;

	system("pause");

	return 0;
}