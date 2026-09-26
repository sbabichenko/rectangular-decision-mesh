// A minimal ordered JSON tree with Python's json.dumps(indent=2) layout and repr(float) number
// text, so a file the port rewrites through json.loads/json.dumps is reproduced byte for byte.
#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace estimator {

std::string repr_double(double value);   // Python's repr(float)

struct JsonValue {
    enum Kind { Null, Bool, Int, Number, String, Array, Object } kind = Null;
    bool boolean = false; long long integer = 0; double number = 0.0; std::string text;
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object;

    static JsonValue null() { return JsonValue{}; }
    static JsonValue of(bool v) { JsonValue j; j.kind = Bool; j.boolean = v; return j; }
    static JsonValue of(int v) { JsonValue j; j.kind = Int; j.integer = v; return j; }
    static JsonValue of(long long v) { JsonValue j; j.kind = Int; j.integer = v; return j; }
    static JsonValue of(double v) { JsonValue j; j.kind = Number; j.number = v; return j; }
    static JsonValue of(const std::string& v) { JsonValue j; j.kind = String; j.text = v; return j; }
    static JsonValue of(const char* v) { return of(std::string(v)); }
    static JsonValue list() { JsonValue j; j.kind = Array; return j; }
    static JsonValue dict() { JsonValue j; j.kind = Object; return j; }
    JsonValue& set(const std::string& key, JsonValue value);   // insert or replace, keeping insertion order
    JsonValue* get(const std::string& key);
    void push(JsonValue value) { array.push_back(std::move(value)); }
    std::string dumps(int indent = 2) const;              // json.dumps(value, indent=indent)
};

}  // namespace estimator
