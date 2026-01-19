#!/bin/bash
set -e

echo "Regenerating C code..."
./sigil2 compile ../src/*.sg > sigil_bootstrap.c 2>&1
echo "✅ Generated $(wc -l < sigil_bootstrap.c) lines"

echo "Applying bug fixes..."

# Fix 1: Remove duplicate sigil_add
sed -i '/^SigilValue sigil_add(SigilValue a, SigilValue b) { return sigil_int/d' sigil_bootstrap.c

# Fix 2: Remove orphan #endif
sed -i '/^#endif \/\* SIGIL_BUILTINS_DEFINED \*\/$/d' sigil_bootstrap.c

# Fix 3: Fix qualify_name
sed -i 's/sigil_qualify_name(/sigil_LoweringContext____qualify_name(/g' sigil_bootstrap.c

# Fix 4: Fix variable redefinitions
sed -i '36662s/SigilValue _ =/SigilValue _unused =/' sigil_bootstrap.c

# Fix 5: Add missing functions
cat > /tmp/missing_impl.c << 'EOF'

/* Missing Runtime Functions */
SigilValue sigil_String____is_empty(SigilValue s) {
    return sigil_bool(s.tag != TAG_STRING || !s.v.s || s.v.s[0] == 0);
}

SigilValue sigil_String____push(SigilValue s, SigilValue ch) {
    if (s.tag != TAG_STRING || ch.tag != TAG_CHAR) return s;
    size_t len = s.v.s ? strlen(s.v.s) : 0;
    char* new_str = (char*)malloc(len + 2);
    if (s.v.s) memcpy(new_str, s.v.s, len);
    new_str[len] = (char)ch.v.i;
    new_str[len + 1] = 0;
    return sigil_string(new_str);
}

SigilValue sigil_String____contains(SigilValue s, SigilValue substr) {
    if (s.tag != TAG_STRING || substr.tag != TAG_STRING) return sigil_bool(false);
    if (!s.v.s || !substr.v.s) return sigil_bool(false);
    return sigil_bool(strstr(s.v.s, substr.v.s) != NULL);
}

SigilValue sigil_String____clone(SigilValue s) {
    if (s.tag != TAG_STRING || !s.v.s) return sigil_string("");
    return sigil_string(s.v.s);
}

SigilValue sigil_Vec____len(SigilValue v) {
    if (v.tag != TAG_ARRAY) return sigil_int(0);
    return sigil_int((long long)v.v.arr.len);
}

SigilValue sigil_Box____into_raw(SigilValue boxed) {
    return boxed;
}

SigilValue sigil_with_note(SigilValue v, SigilValue note) {
    (void)note;
    return v;
}

SigilValue sigil_skip(SigilValue arr, SigilValue n) {
    if (arr.tag != TAG_ARRAY || !arr.v.arr.data) return sigil_array(0);
    size_t skip = (size_t)n.v.i;
    if (skip >= arr.v.arr.len) return sigil_array(0);
    SigilValue result = sigil_array(arr.v.arr.len - skip);
    for (size_t i = 0; i < arr.v.arr.len - skip; i++)
        result.v.arr.data[i] = arr.v.arr.data[i + skip];
    return result;
}

SigilValue sigil_any(SigilValue arr, SigilValue closure) {
    if (arr.tag != TAG_ARRAY || !arr.v.arr.data) return sigil_bool(false);
    typedef SigilValue (*ClosureFn)(SigilValue);
    ClosureFn fn = (ClosureFn)closure.v.ptr;
    for (size_t i = 0; i < arr.v.arr.len; i++) {
        SigilValue result = fn(arr.v.arr.data[i]);
        if (sigil_truthy(result)) return sigil_bool(true);
    }
    return sigil_bool(false);
}
EOF
sed -i '520r /tmp/missing_impl.c' sigil_bootstrap.c

# Fix 6: Add TAG_ENUM case to sigil_display
# Find line with "default: return sigil_string(\"<value>\");"
LINE=$(grep -n 'default: return sigil_string("<value>");' sigil_bootstrap.c | cut -d: -f1)
if [ -n "$LINE" ]; then
    sed -i "${LINE}i\\        case TAG_ENUM: {\\
            if (!v.v.e.data) {\\
                snprintf(buf, sizeof(buf), \"Enum(%u:%u)\", v.v.e.enum_id, v.v.e.variant);\\
            } else {\\
                snprintf(buf, sizeof(buf), \"Enum(%u:%u with %u fields)\", \\
                         v.v.e.enum_id, v.v.e.variant, v.v.e.field_count);\\
            }\\
            break;\\
        }" sigil_bootstrap.c
    echo "✅ Added TAG_ENUM case at line $LINE"
fi

echo "✅ All fixes applied"
wc -l sigil_bootstrap.c
