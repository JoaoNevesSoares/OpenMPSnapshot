#ifndef BOOST_GIL_EXTENSION_DYNAMIC_IMAGE_ANY_IMAGE_HPP
#define BOOST_GIL_EXTENSION_DYNAMIC_IMAGE_ANY_IMAGE_HPP

#include <boost/gil/extension/dynamic_image/any_image_view.hpp>
#include <boost/gil/extension/dynamic_image/apply_operation.hpp>

#include <boost/gil/image.hpp>
#include <boost/gil/detail/mp11.hpp>

#include <boost/config.hpp>
#include <boost/variant2/variant.hpp>

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(push)
#pragma warning(disable:4512) 
#endif

namespace boost { namespace gil {

namespace detail {

template <typename T>
using get_view_t = typename T::view_t;

template <typename Images>
using images_get_views_t = mp11::mp_transform<get_view_t, Images>;

template <typename T>
using get_const_view_t = typename T::const_view_t;

template <typename Images>
using images_get_const_views_t = mp11::mp_transform<get_const_view_t, Images>;

struct recreate_image_fnobj
{
using result_type = void;
point<std::ptrdiff_t> const& _dimensions;
unsigned _alignment;

recreate_image_fnobj(point<std::ptrdiff_t> const& dims, unsigned alignment)
: _dimensions(dims), _alignment(alignment)
{}

template <typename Image>
result_type operator()(Image& img) const { img.recreate(_dimensions,_alignment); }
};

template <typename AnyView>  
struct any_image_get_view
{
using result_type = AnyView;
template <typename Image>
result_type operator()(Image& img) const
{
return result_type(view(img));
}
};

template <typename AnyConstView>  
struct any_image_get_const_view
{
using result_type = AnyConstView;
template <typename Image>
result_type operator()(Image const& img) const { return result_type{const_view(img)}; }
};

} 


template <typename ...Images>
class any_image : public variant2::variant<Images...>
{
using parent_t = variant2::variant<Images...>;
public:    
using view_t = mp11::mp_rename<detail::images_get_views_t<any_image>, any_image_view>;
using const_view_t = mp11::mp_rename<detail::images_get_const_views_t<any_image>, any_image_view>;
using x_coord_t = std::ptrdiff_t;
using y_coord_t = std::ptrdiff_t;
using point_t = point<std::ptrdiff_t>;

any_image() = default;
any_image(any_image const& img) : parent_t((parent_t const&)img) {}

template <typename Image>
explicit any_image(Image const& img) : parent_t(img) {}

template <typename Image>
any_image(Image&& img) : parent_t(std::move(img)) {}

template <typename Image>
explicit any_image(Image& img, bool do_swap) : parent_t(img, do_swap) {}

template <typename ...OtherImages>
any_image(any_image<OtherImages...> const& img)
: parent_t((variant2::variant<OtherImages...> const&)img)
{}

any_image& operator=(any_image const& img)
{
parent_t::operator=((parent_t const&)img);
return *this;
}

template <typename Image>
any_image& operator=(Image const& img)
{
parent_t::operator=(img);
return *this;
}

template <typename ...OtherImages>
any_image& operator=(any_image<OtherImages...> const& img)
{
parent_t::operator=((typename variant2::variant<OtherImages...> const&)img);
return *this;
}

void recreate(const point_t& dims, unsigned alignment=1)
{
apply_operation(*this, detail::recreate_image_fnobj(dims, alignment));
}

void recreate(x_coord_t width, y_coord_t height, unsigned alignment=1)
{
recreate({ width, height }, alignment);
}

std::size_t num_channels() const
{
return apply_operation(*this, detail::any_type_get_num_channels());
}

point_t dimensions() const
{
return apply_operation(*this, detail::any_type_get_dimensions());
}

x_coord_t width()  const { return dimensions().x; }
y_coord_t height() const { return dimensions().y; }
};



template <typename ...Images>
BOOST_FORCEINLINE
auto view(any_image<Images...>& img) -> typename any_image<Images...>::view_t
{
using view_t = typename any_image<Images...>::view_t;
return apply_operation(img, detail::any_image_get_view<view_t>());
}

template <typename ...Images>
BOOST_FORCEINLINE
auto const_view(any_image<Images...> const& img) -> typename any_image<Images...>::const_view_t
{
using view_t = typename any_image<Images...>::const_view_t;
return apply_operation(img, detail::any_image_get_const_view<view_t>());
}

}}  

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(pop)
#endif

#endif