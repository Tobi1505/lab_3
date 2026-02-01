#include <cstdint>
#include <iostream>

int main(int argc, char** argv) {
	// Liste deiner Bilder aus dem input-Ordner
	std::vector<std::string> test_images = {
		"input/test_image_1.bmp",
		"input/test_image_2.bmp",
		"input/test_image_3.bmp",
		"input/test_image_5.bmp"
	};

	std::cout << "--- Performance Benchmark: Sobel Filter ---" << std::endl;

	for (const std::string& filename : test_images) {
		// 1. Bild laden
		BitmapImage bitmap;
		if (!bitmap.load(filename)) {
			std::cerr << "Fehler: Konnte " << filename << " nicht laden!" << std::endl;
			continue;
		}

		// 2. In Graustufen umwandeln
		GrayscaleImage gray;
		gray.load_bitmap(bitmap);

		// 3. IntermediateImage vorbereiten
		IntermediateImage inter(gray.height, gray.width);
		inter.load_grayscale_image(gray);

		// 4. Sobel-Filter ausführen
		// Die Zeitmessung erfolgt direkt IN apply_sobel_filter (Schritt 1 von vorhin)
		inter.apply_sobel_filter();
	}

	std::cout << "--- Benchmark beendet ---" << std::endl;
	return 0;
}
