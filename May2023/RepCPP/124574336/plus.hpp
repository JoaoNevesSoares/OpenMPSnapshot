
#ifndef BOOST_SPIRIT_QI_OPERATOR_PLUS_HPP
#define BOOST_SPIRIT_QI_OPERATOR_PLUS_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/qi/parser.hpp>
#include <boost/spirit/home/support/container.hpp>
#include <boost/spirit/home/qi/detail/attributes.hpp>
#include <boost/spirit/home/qi/detail/fail_function.hpp>
#include <boost/spirit/home/qi/detail/pass_container.hpp>
#include <boost/spirit/home/support/has_semantic_action.hpp>
#include <boost/spirit/home/support/handles_container.hpp>
#include <boost/spirit/home/support/info.hpp>
#include <boost/proto/operators.hpp>
#include <boost/proto/tags.hpp>

namespace boost { namespace spirit
{
template <>
struct use_operator<qi::domain, proto::tag::unary_plus> 
: mpl::true_ {};
}}

namespace boost { namespace spirit { namespace qi
{
template <typename Subject>
struct plus : unary_parser<plus<Subject> >
{
typedef Subject subject_type;

template <typename Context, typename Iterator>
struct attribute
{
typedef typename
traits::build_std_vector<
typename traits::attribute_of<
Subject, Context, Iterator>::type
>::type
type;
};

plus(Subject const& subject_)
: subject(subject_) {}

template <typename F>
bool parse_container(F f) const
{
if (f (subject))
return false;

while (!f (subject))
;
return true;
}

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper
, Attribute& attr_) const
{
typedef detail::fail_function<Iterator, Context, Skipper>
fail_function;

traits::make_container(attr_);

Iterator iter = first;
fail_function f(iter, last, context, skipper);
if (!parse_container(detail::make_pass_container(f, attr_)))
return false;

first = f.first;
return true;
}

template <typename Context>
info what(Context& context) const
{
return info("plus", subject.what(context));
}

Subject subject;
};

template <typename Elements, typename Modifiers>
struct make_composite<proto::tag::unary_plus, Elements, Modifiers>
: make_unary_composite<Elements, plus>
{};
}}}

namespace boost { namespace spirit { namespace traits
{
template <typename Subject>
struct has_semantic_action<qi::plus<Subject> >
: unary_has_semantic_action<Subject> {};

template <typename Subject, typename Attribute, typename Context
, typename Iterator>
struct handles_container<qi::plus<Subject>, Attribute, Context
, Iterator>
: mpl::true_ {}; 
}}}

#endif