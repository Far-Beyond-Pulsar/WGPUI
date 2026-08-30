//! Native colors and retained background descriptions.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, de};

/// A straight-alpha RGBA color with channels in the inclusive range 0..=1.
#[derive(Clone, Copy, Default, PartialEq)]
#[repr(C)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl fmt::Debug for Rgba {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "rgba({:#010x})", u32::from(*self))
    }
}

impl Rgba {
    pub fn opacity(mut self, factor: f32) -> Self {
        self.a *= factor.clamp(0.0, 1.0);
        self
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.a = alpha.clamp(0.0, 1.0);
        self
    }

    pub fn is_transparent(self) -> bool {
        self.a == 0.0
    }

    pub fn is_opaque(self) -> bool {
        self.a == 1.0
    }

    pub fn blend(self, other: Self) -> Self {
        if other.a >= 1.0 {
            other
        } else if other.a <= 0.0 {
            self
        } else {
            Self {
                r: self.r * (1.0 - other.a) + other.r * other.a,
                g: self.g * (1.0 - other.a) + other.g * other.a,
                b: self.b * (1.0 - other.a) + other.b * other.a,
                a: self.a,
            }
        }
    }
}

impl From<Rgba> for [f32; 4] {
    fn from(value: Rgba) -> Self {
        [value.r, value.g, value.b, value.a]
    }
}

impl From<Rgba> for u32 {
    fn from(value: Rgba) -> Self {
        let channel = |channel: f32| (channel.clamp(0.0, 1.0) * 255.0).round() as u32;
        channel(value.r) << 24 | channel(value.g) << 16 | channel(value.b) << 8 | channel(value.a)
    }
}

impl TryFrom<&str> for Rgba {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let hex = value.trim().strip_prefix('#').ok_or_else(|| {
            format!("expected a hex color such as #336699 or #336699cc, got {value:?}")
        })?;
        let digits = hex.as_bytes();
        if !digits.iter().all(u8::is_ascii_hexdigit) {
            return Err(format!(
                "invalid hex color {value:?}; expected hexadecimal digits"
            ));
        }
        let parse_digit = |digit: u8| -> Result<u8, String> {
            match digit {
                b'0'..=b'9' => Ok(digit - b'0'),
                b'a'..=b'f' => Ok(digit - b'a' + 10),
                b'A'..=b'F' => Ok(digit - b'A' + 10),
                _ => Err(format!(
                    "invalid hex color {value:?}; expected hexadecimal digits"
                )),
            }
        };
        let parse_pair = |index: usize| -> Result<u8, String> {
            Ok(parse_digit(digits[index])? * 16 + parse_digit(digits[index + 1])?)
        };
        let parse_short = |index: usize| -> Result<u8, String> {
            let digit = parse_digit(digits[index])?;
            Ok(digit * 16 + digit)
        };
        let (red, green, blue, alpha) = match hex.len() {
            3 | 4 => {
                let red = parse_short(0)?;
                let green = parse_short(1)?;
                let blue = parse_short(2)?;
                let alpha = if hex.len() == 4 { parse_short(3)? } else { 255 };
                (red, green, blue, alpha)
            }
            6 | 8 => {
                let red = parse_pair(0)?;
                let green = parse_pair(2)?;
                let blue = parse_pair(4)?;
                let alpha = if hex.len() == 8 { parse_pair(6)? } else { 255 };
                (red, green, blue, alpha)
            }
            _ => {
                return Err(format!(
                    "invalid hex color {value:?}; expected 3, 4, 6, or 8 digits"
                ));
            }
        };
        Ok(rgb_channels(red, green, blue, alpha))
    }
}

impl Serialize for Rgba {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format!("#{:08x}", u32::from(*self)))
    }
}

impl<'de> Deserialize<'de> for Rgba {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Self::try_from(value.as_str()).map_err(de::Error::custom)
    }
}

/// A hue/saturation/lightness color with normalized channels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct Hsla {
    pub h: f32,
    pub s: f32,
    pub l: f32,
    pub a: f32,
}

impl Hsla {
    pub const fn red() -> Self {
        red()
    }

    pub const fn green() -> Self {
        green()
    }

    pub const fn blue() -> Self {
        blue()
    }

    pub const fn black() -> Self {
        black()
    }

    pub const fn white() -> Self {
        white()
    }

    pub const fn transparent_black() -> Self {
        transparent_black()
    }

    pub fn opacity(mut self, factor: f32) -> Self {
        self.a *= factor.clamp(0.0, 1.0);
        self
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.a = alpha.clamp(0.0, 1.0);
        self
    }

    pub fn fade_out(&mut self, factor: f32) {
        self.a *= 1.0 - factor.clamp(0.0, 1.0);
    }

    pub fn to_rgb(self) -> Rgba {
        self.into()
    }

    pub fn is_transparent(self) -> bool {
        self.a == 0.0
    }

    pub fn is_opaque(self) -> bool {
        self.a == 1.0
    }

    pub fn grayscale(self) -> Self {
        Self { s: 0.0, ..self }
    }

    pub fn blend(self, other: Self) -> Self {
        Hsla::from(Rgba::from(self).blend(Rgba::from(other)))
    }
}

impl From<Hsla> for [f32; 4] {
    fn from(value: Hsla) -> Self {
        Rgba::from(value).into()
    }
}

impl From<Hsla> for Rgba {
    fn from(value: Hsla) -> Self {
        let hue = value.h.rem_euclid(1.0);
        let chroma = (1.0 - (2.0 * value.l - 1.0).abs()) * value.s;
        let second = chroma * (1.0 - ((hue * 6.0).rem_euclid(2.0) - 1.0).abs());
        let match_value = value.l - chroma / 2.0;
        let (red, green, blue) = match (hue * 6.0).floor() as u32 {
            0 => (chroma, second, 0.0),
            1 => (second, chroma, 0.0),
            2 => (0.0, chroma, second),
            3 => (0.0, second, chroma),
            4 => (second, 0.0, chroma),
            _ => (chroma, 0.0, second),
        };
        Rgba {
            r: (red + match_value).clamp(0.0, 1.0),
            g: (green + match_value).clamp(0.0, 1.0),
            b: (blue + match_value).clamp(0.0, 1.0),
            a: value.a.clamp(0.0, 1.0),
        }
    }
}

impl From<Rgba> for Hsla {
    fn from(value: Rgba) -> Self {
        let maximum = value.r.max(value.g).max(value.b);
        let minimum = value.r.min(value.g).min(value.b);
        let lightness = (maximum + minimum) / 2.0;
        let delta = maximum - minimum;
        if delta == 0.0 {
            return hsla(0.0, 0.0, lightness, value.a);
        }
        let saturation = delta / (1.0 - (2.0 * lightness - 1.0).abs());
        let hue = if maximum == value.r {
            ((value.g - value.b) / delta).rem_euclid(6.0) / 6.0
        } else if maximum == value.g {
            ((value.b - value.r) / delta + 2.0) / 6.0
        } else {
            ((value.r - value.g) / delta + 4.0) / 6.0
        };
        hsla(hue, saturation, lightness, value.a)
    }
}

pub fn hsla(h: f32, s: f32, l: f32, a: f32) -> Hsla {
    Hsla {
        h: h.clamp(0.0, 1.0),
        s: s.clamp(0.0, 1.0),
        l: l.clamp(0.0, 1.0),
        a: a.clamp(0.0, 1.0),
    }
}

const fn named(h: f32, s: f32, l: f32) -> Hsla {
    Hsla { h, s, l, a: 1.0 }
}

pub const fn black() -> Hsla {
    named(0.0, 0.0, 0.0)
}
pub const fn white() -> Hsla {
    named(0.0, 0.0, 1.0)
}
pub const fn red() -> Hsla {
    named(0.0, 1.0, 0.5)
}
pub const fn green() -> Hsla {
    named(1.0 / 3.0, 1.0, 0.25)
}
pub const fn blue() -> Hsla {
    named(2.0 / 3.0, 1.0, 0.5)
}
pub const fn yellow() -> Hsla {
    named(1.0 / 6.0, 1.0, 0.5)
}
pub const fn transparent_black() -> Hsla {
    Hsla { a: 0.0, ..black() }
}
pub const fn transparent_white() -> Hsla {
    Hsla { a: 0.0, ..white() }
}

pub fn opaque_grey(lightness: f32, opacity: f32) -> Hsla {
    hsla(0.0, 0.0, lightness, opacity)
}

pub fn rgb(hex: u32) -> Rgba {
    let bytes = hex.to_be_bytes();
    rgb_channels(bytes[1], bytes[2], bytes[3], 255)
}

pub fn rgba(hex: u32) -> Rgba {
    let bytes = hex.to_be_bytes();
    rgb_channels(bytes[0], bytes[1], bytes[2], bytes[3])
}

fn rgb_channels(red: u8, green: u8, blue: u8, alpha: u8) -> Rgba {
    Rgba {
        r: red as f32 / 255.0,
        g: green as f32 / 255.0,
        b: blue as f32 / 255.0,
        a: alpha as f32 / 255.0,
    }
}

/// A color interpolation space retained by gradients.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpace {
    #[default]
    Srgb,
    Oklab,
}

/// A stop in a background gradient.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GradientStop {
    pub color: Hsla,
    pub position: f32,
}

pub fn gradient_color_stop(color: impl Into<Hsla>, position: f32) -> GradientStop {
    GradientStop {
        color: color.into(),
        position,
    }
}

impl GradientStop {
    pub fn opacity(self, alpha: f32) -> Self {
        Self {
            color: self.color.opacity(alpha),
            ..self
        }
    }
}

/// A stop in a text gradient.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LinearColorStop {
    pub color: Hsla,
    pub percentage: f32,
}

pub fn linear_color_stop(color: impl Into<Hsla>, percentage: f32) -> LinearColorStop {
    LinearColorStop {
        color: color.into(),
        percentage,
    }
}

impl LinearColorStop {
    pub fn opacity(self, alpha: f32) -> Self {
        Self {
            color: self.color.opacity(alpha),
            ..self
        }
    }
}

/// A retained background value. The widget layer lowers this description to
/// the existing quad primitive without introducing an immediate paint path.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Background {
    Solid(Hsla),
    LinearGradient {
        angle: f32,
        colors: [GradientStop; 2],
        color_space: ColorSpace,
    },
    RadialGradient {
        center: [f32; 2],
        radius: [f32; 2],
        colors: [GradientStop; 2],
        color_space: ColorSpace,
    },
    PatternSlash {
        color: Hsla,
        width: f32,
        interval: f32,
    },
}

impl Default for Background {
    fn default() -> Self {
        Self::Solid(Hsla::default())
    }
}

impl From<Hsla> for Background {
    fn from(value: Hsla) -> Self {
        Self::Solid(value)
    }
}

impl From<Rgba> for Background {
    fn from(value: Rgba) -> Self {
        Self::Solid(value.into())
    }
}

impl Background {
    pub fn color_space(self, color_space: ColorSpace) -> Self {
        match self {
            Self::LinearGradient { angle, colors, .. } => Self::LinearGradient {
                angle,
                colors,
                color_space,
            },
            Self::RadialGradient {
                center,
                radius,
                colors,
                ..
            } => Self::RadialGradient {
                center,
                radius,
                colors,
                color_space,
            },
            other => other,
        }
    }

    pub fn opacity(self, alpha: f32) -> Self {
        match self {
            Self::Solid(color) => Self::Solid(color.opacity(alpha)),
            Self::LinearGradient {
                angle,
                colors,
                color_space,
            } => Self::LinearGradient {
                angle,
                colors: [colors[0].opacity(alpha), colors[1].opacity(alpha)],
                color_space,
            },
            Self::RadialGradient {
                center,
                radius,
                colors,
                color_space,
            } => Self::RadialGradient {
                center,
                radius,
                colors: [colors[0].opacity(alpha), colors[1].opacity(alpha)],
                color_space,
            },
            Self::PatternSlash {
                color,
                width,
                interval,
            } => Self::PatternSlash {
                color: color.opacity(alpha),
                width,
                interval,
            },
        }
    }

    pub fn is_transparent(self) -> bool {
        match self {
            Self::Solid(color) | Self::PatternSlash { color, .. } => color.is_transparent(),
            Self::LinearGradient { colors, .. } | Self::RadialGradient { colors, .. } => {
                colors.iter().all(|stop| stop.color.is_transparent())
            }
        }
    }
}

pub fn linear_gradient(
    angle: f32,
    from: impl Into<GradientStop>,
    to: impl Into<GradientStop>,
) -> Background {
    Background::LinearGradient {
        angle,
        colors: [from.into(), to.into()],
        color_space: ColorSpace::default(),
    }
}

pub fn radial_gradient(
    center_x: f32,
    center_y: f32,
    radius_x: f32,
    radius_y: f32,
    from: impl Into<GradientStop>,
    to: impl Into<GradientStop>,
) -> Background {
    Background::RadialGradient {
        center: [center_x, center_y],
        radius: [radius_x, radius_y],
        colors: [from.into(), to.into()],
        color_space: ColorSpace::default(),
    }
}

pub fn pattern_slash(color: impl Into<Hsla>, width: f32, interval: f32) -> Background {
    Background::PatternSlash {
        color: color.into(),
        width,
        interval,
    }
}

pub fn solid_background(color: impl Into<Hsla>) -> Background {
    Background::Solid(color.into())
}

#[derive(Clone, Debug)]
pub struct Colors {
    pub text: Rgba,
    pub text_muted: Rgba,
    pub selected_text: Rgba,
    pub background: Rgba,
    pub surface: Rgba,
    pub surface_hover: Rgba,
    pub disabled: Rgba,
    pub selected: Rgba,
    pub border: Rgba,
    pub separator: Rgba,
    pub container: Rgba,
    pub accent: Rgba,
    pub accent_hover: Rgba,
    pub accent_active: Rgba,
    pub success: Rgba,
    pub success_hover: Rgba,
    pub warning: Rgba,
    pub warning_hover: Rgba,
    pub error: Rgba,
    pub error_hover: Rgba,
}

impl Colors {
    pub fn light() -> Self {
        Self {
            text: rgb(0x1d1d1f),
            text_muted: rgb(0x86868b),
            selected_text: rgb(0xffffff),
            background: rgb(0xffffff),
            surface: rgb(0xf5f5f7),
            surface_hover: rgb(0xe8e8ed),
            disabled: rgb(0xb0b0b0),
            selected: rgb(0x0066cc),
            border: rgb(0xd2d2d7),
            separator: rgb(0xd2d2d7),
            container: rgb(0xf5f5f7),
            accent: rgb(0x007aff),
            accent_hover: rgb(0x0071e3),
            accent_active: rgb(0x0058d0),
            success: rgb(0x28cd41),
            success_hover: rgb(0x23b839),
            warning: rgb(0xff9f0a),
            warning_hover: rgb(0xe68f09),
            error: rgb(0xff3b30),
            error_hover: rgb(0xe6352b),
        }
    }

    pub fn dark() -> Self {
        Self {
            text: rgb(0xffffff),
            text_muted: rgb(0x98989d),
            selected_text: rgb(0xffffff),
            background: rgb(0x1e1e1e),
            surface: rgb(0x2d2d2d),
            surface_hover: rgb(0x3d3d3d),
            disabled: rgb(0x565656),
            selected: rgb(0x0058d0),
            border: rgb(0x3d3d3d),
            separator: rgb(0x3d3d3d),
            container: rgb(0x262626),
            accent: rgb(0x0a84ff),
            accent_hover: rgb(0x409cff),
            accent_active: rgb(0x0071e3),
            success: rgb(0x30d158),
            success_hover: rgb(0x28cd52),
            warning: rgb(0xffd60a),
            warning_hover: rgb(0xffcc00),
            error: rgb(0xff453a),
            error_hover: rgb(0xff6961),
        }
    }

    pub fn for_appearance<T>(_window: &T) -> Self {
        Self::light()
    }
}

impl Default for Colors {
    fn default() -> Self {
        Self::light()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_colors_accept_short_and_long_forms() {
        assert_eq!(Rgba::try_from("#abc").expect("valid color"), rgb(0xaabbcc));
        assert_eq!(
            Rgba::try_from("#abcd").expect("valid color"),
            rgba(0xaabbccdd)
        );
        assert_eq!(
            Rgba::try_from("#aabbcc").expect("valid color"),
            rgb(0xaabbcc)
        );
        assert_eq!(
            Rgba::try_from("#aabbccdd").expect("valid color"),
            rgba(0xaabbccdd)
        );
        assert!(Rgba::try_from("#12").is_err());
        assert!(Rgba::try_from("red").is_err());
        assert!(Rgba::try_from("#éééé").is_err());
    }

    #[test]
    fn hsla_round_trips_primary_colors() {
        for color in [red(), green(), blue(), black(), white()] {
            let round_trip = Rgba::from(Hsla::from(Rgba::from(color)));
            assert!((round_trip.r - Rgba::from(color).r).abs() < 0.00001);
            assert!((round_trip.g - Rgba::from(color).g).abs() < 0.00001);
            assert!((round_trip.b - Rgba::from(color).b).abs() < 0.00001);
        }
    }

    #[test]
    fn rgba_serializes_as_canonical_eight_digit_hex() {
        let encoded = serde_json::to_string(&rgba(0x12345678)).expect("serialize color");
        assert_eq!(encoded, "\"#12345678\"");
        let decoded: Rgba = serde_json::from_str(&encoded).expect("deserialize color");
        assert_eq!(decoded, rgba(0x12345678));
    }

    #[test]
    fn opacity_multiplies_and_alpha_replaces() {
        let color = hsla(0.0, 1.0, 0.5, 0.8);
        assert_eq!(color.opacity(0.5).a, 0.4);
        assert_eq!(color.alpha(0.25).a, 0.25);
        assert_eq!(Rgba::from(color).opacity(0.5).a, 0.4);
        assert_eq!(Colors::default().background, rgb(0xffffff));
        assert_eq!(Colors::dark().surface, rgb(0x2d2d2d));
    }

    #[test]
    fn gradient_opacity_and_color_space_are_retained() {
        let gradient = linear_gradient(
            45.0,
            gradient_color_stop(red(), 0.0),
            gradient_color_stop(blue(), 1.0),
        )
        .color_space(ColorSpace::Oklab)
        .opacity(0.5);
        assert!(!gradient.is_transparent());
        assert!(matches!(
            gradient,
            Background::LinearGradient {
                color_space: ColorSpace::Oklab,
                ..
            }
        ));
    }
}
