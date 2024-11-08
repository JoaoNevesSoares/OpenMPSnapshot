
#pragma once

#include "lights/light.h"
#include "objects/object.h"
#include "ray.h"

#include <memory>
#include <vector>

// Represents a 3D scene, containing objects and lights
class Scene {
  std::vector<std::unique_ptr<Object>> objects;
  std::vector<std::unique_ptr<Light>> lights;

  // The rendering distance
  // TODO: perhaps this is more suited for RayTracer?
  float max_dist = inf;
  // The background colour, used when a ray does not intersect any objects
  vec3 background;

public:
  Scene() {}
  explicit Scene(std::vector<std::unique_ptr<Object>> &&objects,
                 std::vector<std::unique_ptr<Light>> &&lights)
      : objects{std::move(objects)}, lights{std::move(lights)} {}
  ~Scene() {}

  void addObject(std::unique_ptr<Object> &&object) {
    objects.push_back(std::move(object));
  }

  void addLight(std::unique_ptr<Light> &&light) {
    lights.push_back(std::move(light));
  }

  // Returns the colour resulting from tracing the specified ray in this scene,
  // with max recursive depth 'depth'
  vec3 intersect(const Ray &ray, size_t depth) const;
};
