#ifndef BRUKER2DSEQ_HPP
#define BRUKER2DSEQ_HPP
#include <string>
#include <map>
#include <sstream>
#include <iterator>
#include <stdint.h>

namespace image
{



namespace io
{

class bruker_info
{
    std::map<std::string,std::string> info;
    /*
    ##$RECO_size=( 2 )
    128 128                 <--info
    */
private:
    void load_info(std::ifstream& in)
    {
        std::string line;
        info.clear();
        while(std::getline(in,line))
        {
            if(line.size() < 4 ||
                    line[0] != '#' ||
                    line[1] != '#' ||
                    line[2] != '$')
                continue;

            std::string::iterator sep = std::find(line.begin(),line.end(),'=');
            if(sep == line.end())
                continue;
            std::string name(line.begin()+3,sep);
            info[name] =
                std::string(sep+1,line.end());
            if(*(sep+1) == '(')
            {
                std::string accumulated_info;
                while(in && in.peek() != '#')
                {
		    std::getline(in,line);
		    if(line[0] == '$')
    			continue;
                    accumulated_info += line;
                    accumulated_info += " ";
                }
                using namespace std;
                accumulated_info.erase(remove(accumulated_info.begin(),accumulated_info.end(),'<'),accumulated_info.end());
                accumulated_info.erase(remove(accumulated_info.begin(),accumulated_info.end(),'>'),accumulated_info.end());
                info[name] = accumulated_info;
            }
        }
    }
public:
    template<typename char_type>
    bool load_from_file(const char_type* file_name)

    {
        std::ifstream info(file_name);
        if(!info)
            return false;
        load_info(info);
        return true;
    }
    const std::string& operator[](const std::string& tag)
    {
        return info[tag];
    }
};


class bruker_2dseq
{
    // the 2dseq data
    std::vector<float> data;

    // image dimension
    unsigned int dim[4];

    // spatial resolution
    float resolution[3];
private:
    std::string tmp;
    std::wstring wtmp;

    bool check_name(const char* filename)
    {
        std::string str = filename;
        if(str.length() < 5)
            return false;
        std::string name(str.end()-5,str.end());
        if(name[0] != '2' || name[1] != 'd' || name[2] != 's' || name[3] != 'e' || name[4] != 'q')
            return false;
        return true;
    }
    bool check_name(const wchar_t* filename)
    {
        std::wstring str = filename;
        if(str.length() < 5)
            return false;
        std::wstring name(str.end()-5,str.end());
        if(name[0] != L'2' || name[1] != L'd' || name[2] != L's' || name[3] != L'e' || name[4] != L'q')
            return false;
        return true;
    }
    const char* load_reco(const char* filename)
    {
        std::string str = filename;
        tmp = std::string(str.begin(),str.end()-5);
        tmp += "reco";
        return tmp.c_str();
    }
    const wchar_t* load_reco(const wchar_t* filename)
    {
        std::wstring str = filename;
        wtmp = std::wstring(str.begin(),str.end()-5);
        wtmp += L"reco";
        return wtmp.c_str();
    }
    const char* load_visu(const char* filename)
    {
        std::string str = filename;
        tmp = std::string(str.begin(),str.end()-5);
        tmp += "visu_pars";
        return tmp.c_str();
    }
    const wchar_t* load_visu(const wchar_t* filename)
    {
        std::wstring str = filename;
        wtmp = std::wstring(str.begin(),str.end()-5);
        wtmp += L"visu_pars";
        return wtmp.c_str();
    }
    
public:
    

    template<typename char_type>
    bool load_from_file(const char_type* file_name)
    {
        if(!check_name(file_name))
            return false;

        // read image dimension
        bruker_info visu,info;
        if(!visu.load_from_file(load_visu(file_name)) ||
           !info.load_from_file(load_reco(file_name)) )
            return false;

        // get image dimension
        std::fill(dim,dim+4,1);
        std::istringstream(visu["VisuCoreSize"]) >> dim[0] >> dim[1] >> dim[2];



        std::vector<char> buffer;
        std::ifstream in(file_name,std::ios::binary);
        in.seekg(0, std::ifstream::end);
        buffer.resize(in.tellg());
        in.seekg(0, std::ifstream::beg);
        in.read((char*)&*buffer.begin(),buffer.size());

        int word_size = 1;
        if (info["RECO_wordtype"].find("16BIT") != std::string::npos)
            word_size = 2;
        if (info["RECO_wordtype"].find("32BIT") != std::string::npos)
            word_size = 4;
        if(info["RECO_byte_order"] == std::string("bigEndian"))
        {
            if (word_size == 2)
                change_endian((short*)&buffer[0],buffer.size()/word_size);
            if (word_size == 4)
                change_endian((int*)&buffer[0],buffer.size()/word_size);
        }
        // read 2dseq and convert to float
        data.resize(buffer.size()/word_size);
        if (info["RECO_wordtype"] == std::string("_8BIT_SGN_INT"))
            std::copy((char*)&buffer[0],(char*)&buffer[0]+data.size(),data.begin());
        if (info["RECO_wordtype"] == std::string("_8BIT_USGN_INT"))
            std::copy((unsigned char*)&buffer[0],(unsigned char*)&buffer[0]+data.size(),data.begin());
        if (info["RECO_wordtype"] == std::string("_16BIT_SGN_INT"))
            std::copy((int16_t*)&buffer[0],(int16_t*)&buffer[0]+data.size(),data.begin());
        if (info["RECO_wordtype"] == std::string("_16BIT_USGN_INT"))
            std::copy((uint16_t*)&buffer[0],(uint16_t*)&buffer[0]+data.size(),data.begin());
        if (info["RECO_wordtype"] == std::string("_32BIT_USGN_INT"))
            std::copy((uint32_t*)&buffer[0],(uint32_t*)&buffer[0]+data.size(),data.begin());
        if (info["RECO_wordtype"] == std::string("_32BIT_SGN_INT"))
            std::copy((int32_t*)&buffer[0],(int32_t*)&buffer[0]+data.size(),data.begin());
        if (info["RECO_wordtype"] == std::string("_32BIT_FLOAT"))
            std::copy((float*)&buffer[0],(float*)&buffer[0]+data.size(),data.begin());
        if(dim[2] == 1)
            dim[2] = data.size()/dim[0]/dim[1];
        if(dim[3] == 1)
            dim[3] = data.size()/dim[0]/dim[1]/dim[2];

        // get resolution
        {
            std::vector<float> fov_data; // in cm
            std::istringstream fov_text(info["RECO_fov"]);
            std::copy(std::istream_iterator<float>(fov_text),
                      std::istream_iterator<float>(),
                      std::back_inserter(fov_data));
            std::fill(resolution,resolution+3,0.0);
            for(unsigned int index = 0;index < 3 && index < fov_data.size();++index)
                resolution[index] = fov_data[index]*10.0/(float)dim[index]; // in mm
        }
        std::vector<float> slopes;
        {
            std::istringstream slope_text_parser(info["RECO_map_slope"]);
            std::copy(std::istream_iterator<double>(slope_text_parser),
                      std::istream_iterator<double>(),
                      std::back_inserter(slopes));
            float max_slope = *std::max_element(slopes.begin(),slopes.end());
            for(unsigned int i = 0;i < slopes.size();++i)
                slopes[i] /= max_slope;
        }
        if(!slopes.empty())
        {
            unsigned int plane_size = dim[0]*dim[1];
            std::vector<float>::iterator iter = data.begin();
            for(unsigned int z = 0;z < dim[2];++z)
            {
                int slope_index = std::floor(float(z)*slopes.size()/dim[2]);
                if(slope_index >= slopes.size())
                   slope_index = slopes.size()-1;
                float s = slopes[slope_index];
                for(unsigned int index = 0;index < plane_size;++index,++iter)
                    *iter /= s;
            }
        }
        return true;

    }

    template<typename pixel_size_type>
    void get_voxel_size(pixel_size_type pixel_size_from) const
    {
        if(dim[2] >= 1)
            std::copy(resolution,resolution+3,pixel_size_from);
        else
            std::copy(resolution,resolution+2,pixel_size_from);
    }

    template<typename image_type>
    void save_to_image(image_type& out) const
    {
        out.resize(geometry<image_type::dimension>(dim));
        std::copy(data.begin(),data.begin()+out.size(),out.begin());
    }

    template<typename image_type>
    const bruker_2dseq& operator>>(image_type& source) const
    {
        save_to_image(source);
        return *this;
    }
};




}






}





#endif//2DSEQ_HPP
