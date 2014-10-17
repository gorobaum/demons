#ifndef optimization_hpp
#define optimization_hpp

#include <limits>
#include <vector>
#include <map>
#include "image/numerical/numerical.hpp"
#include "image/numerical/matrix.hpp"
namespace image
{

namespace optimization
{

template<typename image_type,typename iter_type1,typename function_type>
void plot_fun_2d(
                image_type& I,
                iter_type1 x_beg,iter_type1 x_end,
                iter_type1 x_upper,iter_type1 x_lower,
                function_type& fun,
                unsigned int dim1,unsigned int dim2,unsigned int sample_frequency = 100)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    I.resize(image::geometry<2>(sample_frequency,sample_frequency));
    for(image::pixel_index<2> index;index.is_valid(I.geometry());index.next(I.geometry()))
    {
        std::vector<param_type> x(x_beg,x_end);
        x[dim1] = (x_upper[dim1]-x_lower[dim1])*index[0]/(float)sample_frequency+x_lower[dim1];
        x[dim2] = (x_upper[dim2]-x_lower[dim2])*index[1]/(float)sample_frequency+x_lower[dim2];
        I[index.index()] = fun(x.begin());
    }
}

// calculate fun(x+ei)
template<typename iter_type1,typename tol_type,typename iter_type2,typename function_type>
void estimate_change(iter_type1 x_beg,iter_type1 x_end,tol_type tol,iter_type2 fun_ei,function_type& fun)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    unsigned int size = x_end-x_beg;
    for(unsigned int i = 0;i < size;++i)
    {
        if(tol[i] == 0)
            continue;
        param_type old_x = x_beg[i];
        x_beg[i] += tol[i];
        fun_ei[i] = fun(x_beg);
        x_beg[i] = old_x;
    }
}
// calculate fun(x+ei)
template<typename storage_type,typename tol_storage_type,typename fun_type,typename function_type>
void estimate_change(const storage_type& x,const tol_storage_type& tol,fun_type& fun_ei,function_type& fun)
{
    estimate_change(x.begin(),x.end(),tol.begin(),fun_ei.begin(),fun);
}

template<typename iter_type1,typename tol_type,typename value_type,typename iter_type2,typename iter_type3>
void gradient(iter_type1 x_beg,iter_type1 x_end,
              tol_type tol,
              value_type fun_x,
              iter_type2 fun_x_ei,
              iter_type3 g_beg)
{
    unsigned int size = x_end-x_beg;
    std::copy(fun_x_ei,fun_x_ei+size,g_beg);
    image::minus_constant(g_beg,g_beg+size,fun_x);
    for(unsigned int i = 0;i < size;++i)
        if(tol[i] == 0)
            g_beg[i] = 0;
        else
            g_beg[i] /= tol[i];
}
template<typename storage_type,typename tol_storage_type,typename value_type,typename storage_type2,typename storage_type3>
void gradient(const storage_type& x,const tol_storage_type& tol,value_type fun_x,const storage_type2& fun_x_ei,storage_type3& g)
{
    gradient(x.begin(),x.end(),tol.begin(),fun_x,fun_x_ei.begin(),g.begin());
}

template<typename iter_type1,typename tol_type,typename value_type,typename iter_type2,typename iter_type3,typename function_type>
void hessian(iter_type1 x_beg,iter_type1 x_end,
             tol_type tol,
             value_type fun_x,
             iter_type2 fun_x_ei,
             iter_type3 h_iter,
             function_type& fun)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    unsigned int size = x_end-x_beg;
    std::vector<param_type> old_x(x_beg,x_end);
    // h = fun(x+ei+ej)+fun(x)-fun(ei)-fun(ej)
    for(unsigned int i = 0; i < size;++i)
    for(unsigned int j = 0,shift = 0; j < size;++j,++h_iter)
    {
        if(j < i)
            continue;
        param_type tol2 =  tol[i]*tol[j];
        x_beg[i] += tol[i];
        x_beg[j] += tol[j];
        if(tol2 == 0)
            *h_iter = (i == j ? 1.0:0.0);
        else
            *h_iter = (fun(x_beg)-fun_x_ei[i]-fun_x_ei[j]+fun_x)/tol2;
        if(j != i)
            *(h_iter + shift) = *h_iter;
        x_beg[i] = old_x[i];
        x_beg[j] = old_x[j];
        shift += (size-1);
    }
}

template<typename storage_type,typename tol_storage_type,typename value_type,typename storage_type2,typename storage_type3,typename function_type>
void hessian(const storage_type& x,const tol_storage_type& tol,value_type fun_x,const storage_type2& fun_x_ei,storage_type3& h,function_type& fun)
{
    hessian(x.begin(),x.end(),tol.begin(),fun_x,fun_x_ei.begin(),h.begin(),fun);
}

template<typename iter_type1,typename iter_type2,typename g_type,typename value_type,typename function_type>
bool armijo_line_search(iter_type1 x_beg,iter_type1 x_end,
                        iter_type2 x_upper,iter_type2 x_lower,
                        g_type g_beg,
                        value_type& fun_x,
                        function_type& fun,double precision = 0.001)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    unsigned int size = x_end-x_beg;
    double norm = image::norm2(g_beg,g_beg+size);
    for(double step = 1.0;step > precision;step *= 0.5)
    {
        std::vector<param_type> new_x(x_beg,x_end);
        image::vec::aypx(g_beg,g_beg+size,-step,new_x.begin());
        for(unsigned int j = 0;j < size;++j)
            new_x[j] = std::min(std::max(new_x[j],x_lower[j]),x_upper[j]);
        value_type new_fun_x(fun(new_x.begin()));
        if(fun_x-new_fun_x >= 0.0001*step*norm)
        {
            fun_x = new_fun_x;
            std::copy(new_x.begin(),new_x.end(),x_beg);
            //std::cout << fun_x << std::endl;
            return true;
        }
    }
    return false;
}
template<typename storage_type,typename g_storage_type,typename value_type,typename function_type>
bool armijo_line_search(storage_type& x,
                        const storage_type& upper,
                        const storage_type& lower,
                        const g_storage_type& g_beg,
                        value_type& fun_x,
                        function_type& fun,double precision = 0.001)
{
    armijo_line_search(x.begin(),x.end(),upper.begin(),lower.begin(),g_beg.begin(),fun_x,fun,precision);
}

template<typename tol_type,typename iter_type>
double calculate_resolution(tol_type& tols,iter_type x_upper,iter_type x_lower,double precision = 0.001)
{
    for(unsigned int i = 0;i < tols.size();++i)
        tols[i] = (x_upper[i]-x_lower[i])*precision;
    return image::norm2(tols.begin(),tols.end());
}

template<typename iter_type1,typename function_type,typename terminated_class>
void quasi_newtons_minimize(
                iter_type1 x_beg,iter_type1 x_end,
                iter_type1 x_upper,iter_type1 x_lower,
                function_type& fun,
                terminated_class& terminated,double precision = 0.001)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    typedef typename function_type::value_type value_type;
    unsigned int size = x_end-x_beg;
    value_type fun_x(fun(x_beg));
    std::vector<param_type> tols(size);
    double tol_length = calculate_resolution(tols,x_upper,x_lower,precision);
    for(unsigned int iter = 0;iter < 500 && !terminated;++iter)
    {
        std::vector<value_type> fun_x_ei(size);
        std::vector<param_type> g(size),h(size*size),p(size);
        estimate_change(x_beg,x_end,tols.begin(),fun_x_ei.begin(),fun);
        gradient(x_beg,x_end,tols.begin(),fun_x,fun_x_ei.begin(),g.begin());
        hessian(x_beg,x_end,tols.begin(),fun_x,fun_x_ei.begin(),h.begin(),fun);

        // regularize hessian
        //image::normalize(h,1.0);
        //for(unsigned int i = 0;i < size;++i)
        //    h[i + i*size] += 1.0;

        std::vector<unsigned int> pivot(size);
        image::matrix::lu_decomposition(h.begin(),pivot.begin(),image::dyndim(size,size));
        if(!image::matrix::lu_solve(h.begin(),pivot.begin(),g.begin(),p.begin(),image::dyndim(size,size)))
            return;

        image::multiply(p,tols); // scale the unit to parameter unit
        double length = image::norm2(p.begin(),p.end());
        image::multiply_constant(p,tol_length/length);
        if(!armijo_line_search(x_beg,x_end,x_upper,x_lower,p.begin(),fun_x,fun,precision))
            return;
    }
}
template<typename iter_type1,typename iter_type2,typename function_type,typename terminated_class>
void graient_descent(
                iter_type1 x_beg,iter_type1 x_end,
                iter_type2 x_upper,iter_type2 x_lower,
                function_type& fun,
                typename function_type::value_type& fun_x,
                terminated_class& terminated,double precision = 0.001)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    typedef typename function_type::value_type value_type;
    unsigned int size = x_end-x_beg;
    std::vector<param_type> tols(size);
    double tol_length = calculate_resolution(tols,x_upper,x_lower,precision);
    for(unsigned int iter = 0;iter < 1000 && !terminated;++iter)
    {
        std::vector<value_type> fun_x_ei(size);
        std::vector<param_type> g(size);
        estimate_change(x_beg,x_end,tols.begin(),fun_x_ei.begin(),fun);
        gradient(x_beg,x_end,tols.begin(),fun_x,fun_x_ei.begin(),g.begin());

        image::multiply(g,tols); // scale the unit to parameter unit
        double length = image::norm2(g.begin(),g.end());
        image::multiply_constant(g,tol_length/length);
        if(!armijo_line_search(x_beg,x_end,x_upper,x_lower,g.begin(),fun_x,fun,precision))
            return;
    }
}


template<typename iter_type1,typename iter_type2,typename function_type,typename terminated_class>
void conjugate_descent(
                iter_type1 x_beg,iter_type1 x_end,
                iter_type2 x_upper,iter_type2 x_lower,
                function_type& fun,
                typename function_type::value_type& fun_x,
                terminated_class& terminated,double precision = 0.001)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    typedef typename function_type::value_type value_type;
    unsigned int size = x_end-x_beg;
    std::vector<param_type> tols(size);
    double tol_length = calculate_resolution(tols,x_upper,x_lower,precision);

    std::vector<param_type> g(size),d(size),y(size);
    for(unsigned int iter = 0;iter < 1000 && !terminated;++iter)
    {
        std::vector<value_type> fun_x_ei(size);
        estimate_change(x_beg,x_end,tols.begin(),fun_x_ei.begin(),fun);
        gradient(x_beg,x_end,tols.begin(),fun_x,fun_x_ei.begin(),g.begin());
        if(iter == 0)
            d = g;
        else
        {
            image::minus(y.begin(),y.end(),g.begin());      // y = g_k-g_k_1
            double dt_yk = image::vec::dot(d.begin(),d.end(),y.begin());
            double y2 = image::vec::dot(y.begin(),y.end(),y.begin());
            image::vec::axpy(y.begin(),y.end(),-2.0*y2/dt_yk,d.begin()); // y = yk-(2|y|^2/dt_yk)dk
            double beta = image::vec::dot(y.begin(),y.end(),g.begin())/dt_yk;
            image::multiply_constant(d.begin(),d.end(),-beta);
            image::add(d,g);
        }
        y.swap(g);
        g = d;

        image::multiply(g,tols); // scale the unit to parameter unit
        double length = image::norm2(g.begin(),g.end());
        image::multiply_constant(g,tol_length/length);
        if(!armijo_line_search(x_beg,x_end,x_upper,x_lower,g.begin(),fun_x,fun,precision))
            return;
    }
}


template<typename iter_type1,typename iter_type2,typename value_type,typename function_type>
bool rand_search(iter_type1 x_beg,iter_type1 x_end,iter_type2 x_upper,iter_type2 x_lower,
                 value_type& fun_x,function_type& fun,double variance)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    {
        std::vector<param_type> new_x(x_beg,x_end);
        unsigned int size = x_end-x_beg;
        {
            unsigned int iter = 0;
            unsigned int i;
            do
            {
                i = std::rand()%size;
                ++iter;
            }
            while(x_upper[i] == x_lower[i] && iter < 100);
            float seed1 = (float)std::rand()+1.0;
            float seed2 = (float)std::rand()+1.0;
            seed1 /= (float)RAND_MAX+1.0;
            seed2 /= (float)RAND_MAX+1.0;
            seed1 *= 6.28318530718;
            seed2 = std::sqrt(std::max<float>(0.0,-2.0*std::log(seed2)));
            float r1 = seed2*std::cos(seed1);
            new_x[i] += (x_upper[i]-x_lower[i])*r1/variance;
            new_x[i] = std::min(std::max(new_x[i],x_lower[i]),x_upper[i]);
        }
        value_type new_fun_x(fun(new_x.begin()));
        if(new_fun_x < fun_x)
        {
            fun_x = new_fun_x;
            std::copy(new_x.begin(),new_x.end(),x_beg);
            //std::cout << fun_x << std::endl;
            return true;
        }
    }
    return false;
}


template<typename param_type,typename value_type,unsigned int max_iteration = 100>
struct brent_method
{
    param_type min;
    param_type max;
    bool ended;
public:
    brent_method(void):ended(false) {}
    template<typename eval_fun_type,typename termination_type>
    value_type minimize(eval_fun_type& f,value_type& arg_min,termination_type& terminated,value_type tol)
    {
        value_type bx = arg_min;
        value_type a = min;
        value_type b = max;
        struct assign
        {
            void operator()(std::pair<value_type,value_type>& lhs,const std::pair<value_type,value_type>& rhs)
            {
                lhs.first = rhs.first;
                lhs.second = rhs.second;
            }
        };
        std::map<value_type,value_type> record;
        const value_type gold_ratio=0.3819660;
        const value_type ZEPS=std::numeric_limits<double>::epsilon()*1.0e-3;
        value_type d=0.0,e=0.0;
        value_type etemp = f(bx);
        value_type tol1,tol2,xm;

        record[bx] = etemp;

        std::pair<value_type,value_type> x(bx,etemp),w(bx,etemp),v(bx,etemp),u(0.0,0.0);

        for (unsigned int iter=0;iter< max_iteration && !terminated;iter++)
        {
            xm=(a+b)/2.0;
            tol2=2.0*(tol1=tol*std::abs(x.first)+ZEPS);
            if (std::abs(x.first-xm) <= (tol2-0.5*(b-a)))
            {
                goto end;
            }
            if (std::abs(e) > tol1)
            {
                value_type r=(x.first-w.first)*(x.second-v.second);
                value_type q=(x.first-v.first)*(x.second-w.second);
                value_type p=(x.first-v.first)*q-(x.first-w.first)*r;
                q=2.0*(q-r);
                if (q > 0.0)
                    p = -p;
                if (q < 0.0)
                    q = -q;
                etemp=e;
                e=d;
                if (std::abs(p) >= std::abs(0.5*q*etemp) || p <= q*(a-x.first) || p >= q*(b-x.first))
                    d=gold_ratio*(e=(x.first >= xm ? a-x.first : b-x.first));
                else
                {
                    d=p/q;
                    u.first=x.first+d;
                    if (u.first-a < tol2 || b-u.first < tol2)
                        d=tol1 >= 0 ? xm-x.first:x.first-xm;
                }
            }
            else
                d=gold_ratio*(e=(x.first >= xm ? a-x.first : b-x.first));
            u.first=(std::abs(d) >= tol1 ? x.first + d : (x.first + (d >= 0) ? tol1:-tol1));

            typename std::map<value_type,value_type>::const_iterator past_result = record.find(u.first);
            if (past_result != record.end())
                u.second=past_result->second;
            else
            {
                u.second=f(u.first);
                record[u.first] = u.second;
            }
            if (u.second <= x.second)
            {
                if (u.first >= x.first)
                    a=x.first;
                else
                    b=x.first;
                assign()(v,w);
                assign()(w,x);
                assign()(x,u);
            }
            else
            {
                if (u.first < x.first)
                    a=u.first;
                else
                    b=u.first;
                if (u.second <= w.second || w.first == x.first)
                {
                    assign()(v,w);
                    assign()(w,u);
                }
                else
                    if (u.second <= v.second || v.first == x.first || v.first == w.first)
                        assign()(v,u);
            }
        }
end:
        arg_min = x.first;
        ended = true;
        return x.second;
    }
};


template<typename param_type,typename value_type,unsigned int max_iteration = 100>
struct enhanced_brent{
    param_type min;
    param_type max;
    bool ended;
public:
    enhanced_brent(void):ended(false) {}
    template<typename eval_fun_type,typename termination_type>
    value_type minimize(eval_fun_type& f,value_type& out_arg_min,termination_type& terminated,value_type tol)
    {
        param_type cur_min = min;
        param_type cur_max = max;
        param_type arg_min = out_arg_min;
        if(arg_min < min && arg_min > max)
            arg_min = (max+min)/2.0;
        for(unsigned int iter = 0;iter < max_iteration && !terminated;++iter)
        {
            std::deque<value_type> values;
            std::deque<param_type> params;
            param_type interval = (cur_max-cur_min)/10.0;
            for(param_type x = arg_min;x > cur_min;x -= interval)
            {
                values.push_front(f(x));
                params.push_front(x);
            }
            for(param_type x = arg_min+interval;x < cur_max;x += interval)
            {
                values.push_back(f(x));
                params.push_back(x);
            }
            values.push_front(f(cur_min));
            params.push_front(cur_min);
            values.push_back(f(cur_max));
            params.push_back(cur_max);
            std::vector<unsigned char> greater(values.size()-1);
            for(int i=0;i < greater.size();++i)
                greater[i] = values[i] > values[i+1];
            unsigned char change_sign = 0;
            for(int i=1;i < greater.size();++i)
                if(greater[i-1] != greater[i])
                    change_sign++;

            int min_index = std::min_element(values.begin(),values.end())-values.begin();

            cur_min = params[std::max<int>(0,min_index-2)];
            cur_max = params[std::min<int>(params.size()-1,min_index+2)];
            arg_min = params[min_index];
            if(change_sign <= 2) // monotonic or u-shape then use breant method
                break;
        }

        float result = 0.0;
        brent_method<param_type,value_type,max_iteration> brent;
        brent.min = cur_min;
        brent.max = cur_max;
        result = brent.minimize(f,arg_min,terminated,tol);
        ended = true;
        out_arg_min = arg_min;
        return result;
    }

};

/**

    param_type::dimension
	param_type::operator[]

	eval_fun_type::operator()(parameter_type)

*/
template<typename method_type,typename param_type_,typename value_type_,unsigned int max_iteration = 100>
struct powell_method
{
public:
    typedef param_type_ param_type;
    typedef value_type_ value_type;
    std::vector<method_type> search_methods;
    bool ended;
public:
    powell_method(unsigned int dimension):search_methods(dimension),ended(false) {}

    template<typename eval_fun_type,typename value_type>
    struct powell_fasade
    {
        eval_fun_type& eval_fun;
        param_type param;
        unsigned int current_dim;
public:
        powell_fasade(eval_fun_type& eval_fun_,param_type param_,unsigned int current_dim_):
                eval_fun(eval_fun_),param(param_),current_dim(current_dim_) {}

        template<typename input_param_type>
        value_type operator()(input_param_type next_param)
        {
            param[current_dim] = next_param;
            return eval_fun(param);
        }
    };



    template<typename eval_fun_type,typename teminated_class>
    value_type minimize(eval_fun_type& fun,param_type& arg_min,teminated_class& terminated,value_type tol = 0.01)
    {
        // estimate the acceptable error level
        std::vector<value_type> eplson(search_methods.size());
        for (unsigned int j = 0; j < search_methods.size();++j)
            eplson[j] = tol*0.05*(search_methods[j].max - search_methods[j].min);

        value_type min_value = 0;
        bool improved = true;
        powell_fasade<eval_fun_type,value_type> search_fun(fun,arg_min,0);
        for (unsigned int i = 0; i < max_iteration && improved && !terminated;++i)
        {
            improved = false;
            for (unsigned int j = 0; j < search_methods.size() && !terminated;++j)
            {
                search_fun.current_dim = j;
                search_fun.param[j] = arg_min[j];
                if (search_methods[j].min >= search_methods[j].max)
                    continue;
                value_type next_value = search_methods[j].minimize(search_fun,search_fun.param[j],terminated,tol);
                if (!improved && next_value != min_value && std::abs(arg_min[j] - search_fun.param[j]) > eplson[j])
                    improved = true;
                arg_min[j] = search_fun.param[j];
                min_value = next_value;
            }
        }
        ended = true;
        return min_value;
    }

};
/*
template<typename param_type,typename value_type>
struct BFGS
{
    unsigned int dimension;
	unsigned int dim2;
    BFGS(unsigned int dim):dimension(dim),dim2(dim*dim) {}
    template<typename function_type,typename gradient_function_type>
    value_type minimize(const function_type& f,
						const gradient_function_type& g,
						param_type& xk,
						value_type radius,
						value_type tol = 0.001)
    {
        param_type g_k = g(xk);
        param_type p = -g_k;
        std::vector<value_type> invB(dim2),B1(dim2),B2(dim2),B2syn(dim2);
        math::matrix_identity(invB.begin(),math::dyndim(dimension,dimension));
        value_type end_gradient = tol*tol*(g_k*g_k);
		// parameter for back tracking
		value_type line_search_rate = 0.5;
		value_type c1 = 0.0001;
        radius /= std::sqrt(p*p);
        for(unsigned int iter = 0;iter < 100;++iter)
        {
            // back_tracking
            value_type f_x0 = f(xk);
			value_type dir_g_x0 = p*g_k;
			value_type alpha_k = radius;
	        do//back tracking
            {
				param_type x_alpha_dir = p;
				x_alpha_dir *= alpha_k;
				x_alpha_dir += xk;
				// the Armijo rule
				if (f(x_alpha_dir) <= f_x0 + c1*alpha_k*dir_g_x0)
					break;
                alpha_k *= line_search_rate;
            }
            while (alpha_k > 0.0);
			// set Sk = alphak*p;
            param_type s_k = p;s_k *= alpha_k; 
            
			// update Xk <- Xk + s_k
			param_type x_k_1 = xk;x_k_1 += s_k; 
            
			// Yk = g(Xk+1) - g(Xk)
			param_type g_k_1 = g(x_k_1);
            param_type y_k = g_k_1;y_k -= g_k;  
            
			value_type s_k_y_k = s_k*y_k;
			
			if(s_k_y_k == 0.0) // y_k = 0  or alpha too small
				break;

            param_type invB_y_k;
			
			// invB*Yk
            math::matrix_vector_product(invB.begin(),y_k.begin(),invB_y_k.begin(),math::dyndim(dimension,dimension));

			// B1 = Sk*Skt
            math::vector_op_gen(s_k.begin(),s_k.begin()+dimension,s_k.begin(),B1.begin());

			// B2 = B-1YkSkt
            math::vector_op_gen(invB_y_k.begin(),invB_y_k.begin()+dimension,s_k.begin(),B2.begin());

            math::matrix_transpose(B2.begin(),B2.begin(),math::dyndim(dimension,dimension));
			
            double tmp = (s_k_y_k+y_k*invB_y_k)/s_k_y_k;
            for (unsigned int index = 0;index < invB.size();++index)
                invB[index] += (tmp*B1[index]-(B2[index]+B2syn[index]))/s_k_y_k;

            param_type p_k_1;
            math::matrix_vector_product(invB.begin(),g_k_1.begin(),p_k_1.begin(),math::dyndim(dimension,dimension));

            p = -p_k_1;
            xk = x_k_1;
            g_k = g_k_1;
            if (g_k*g_k < end_gradient)
                break;
        }
        return f(xk);
    }

};
*/



}

}

#endif//optimization_hpp
