#include "estimator/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace estimator {

std::string repr_double(double value) {
    if (std::isnan(value)) return "NaN";
    if (std::isinf(value)) return value > 0 ? "Infinity" : "-Infinity";
    if (value == 0.0) return std::signbit(value) ? "-0.0" : "0.0";
    // the shortest digit string that round-trips, then Python's layout: fixed notation for
    // decimal exponents in [-4, 16), otherwise scientific with at least two exponent digits
    char buffer[40];
    int precision = 1;
    for (; precision <= 17; ++precision) {
        std::snprintf(buffer, sizeof buffer, "%.*e", precision - 1, value);
        if (std::strtod(buffer, nullptr) == value) break;
    }
    // buffer: [-]d.ddddde[+-]XX
    std::string s(buffer);
    const bool negative = s[0] == '-';
    if (negative) s.erase(0, 1);
    const std::size_t epos = s.find('e');
    std::string mantissa = s.substr(0, epos);
    const int exponent = std::atoi(s.c_str() + epos + 1);
    std::string digits;
    for (char c : mantissa) if (c != '.') digits += c;
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    std::string out;
    if (exponent >= -4 && exponent < 16) {
        if (exponent >= 0) {
            if (static_cast<int>(digits.size()) <= exponent + 1) {
                out = digits + std::string(exponent + 1 - digits.size(), '0') + ".0";
            } else {
                out = digits.substr(0, exponent + 1) + "." + digits.substr(exponent + 1);
            }
        } else {
            out = "0." + std::string(-exponent - 1, '0') + digits;
        }
    } else {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char e[8]; std::snprintf(e, sizeof e, "e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent));
        out += e;
    }
    return negative ? "-" + out : out;
}

JsonValue& JsonValue::set(const std::string& key, JsonValue value) {
    kind = Object;
    for (auto& [k, v] : object) if (k == key) { v = std::move(value); return v; }
    object.emplace_back(key, std::move(value));
    return object.back().second;
}

JsonValue* JsonValue::get(const std::string& key) {
    for (auto& [k, v] : object) if (k == key) return &v;
    return nullptr;
}

namespace {
std::string quoted(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20 || c >= 0x7f) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
                else out += static_cast<char>(c);
        }
    }
    return out + "\"";
}

void dump(const JsonValue& v, int indent, int level, std::string& out) {
    const std::string pad(indent * (level + 1), ' '), close(indent * level, ' ');
    switch (v.kind) {
        case JsonValue::Null: out += "null"; break;
        case JsonValue::Bool: out += v.boolean ? "true" : "false"; break;
        case JsonValue::Int: out += std::to_string(v.integer); break;
        case JsonValue::Number: out += repr_double(v.number); break;
        case JsonValue::String: out += quoted(v.text); break;
        case JsonValue::Array:
            if (v.array.empty()) { out += "[]"; break; }
            out += "[\n";
            for (std::size_t i = 0; i < v.array.size(); ++i) { out += pad; dump(v.array[i], indent, level + 1, out); out += (i + 1 < v.array.size()) ? ",\n" : "\n"; }
            out += close + "]";
            break;
        case JsonValue::Object:
            if (v.object.empty()) { out += "{}"; break; }
            out += "{\n";
            for (std::size_t i = 0; i < v.object.size(); ++i) {
                out += pad + quoted(v.object[i].first) + ": ";
                dump(v.object[i].second, indent, level + 1, out);
                out += (i + 1 < v.object.size()) ? ",\n" : "\n";
            }
            out += close + "}";
            break;
    }
}
}  // namespace

std::string JsonValue::dumps(int indent) const {
    std::string out;
    dump(*this, indent, 0, out);
    return out;
}

}  // namespace estimator
