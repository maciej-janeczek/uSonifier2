#pragma once

#include <cstdlib>

namespace stereo{
    class Size2d {
        private:
            unsigned int width;
            unsigned int height;
            unsigned int channels;

        public:
            Size2d(unsigned int width, unsigned int height, unsigned int channels) : width(width), height(height), channels(channels) {}

            size_t getSize() const { return width * height * channels; }

            unsigned int getWidth() const { return width; }

            unsigned int getHeight() const { return height; }

            unsigned int getChannels() const { return channels; }
    };
}
